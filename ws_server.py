import os
import json
import asyncio
import uuid
from datetime import datetime
from typing import Optional
import threading
import queue as pyqueue
import time as _time

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse, HTMLResponse
from google.cloud import speech
from pydub import AudioSegment, silence as pydub_silence
from dotenv import load_dotenv

load_dotenv()

def trim_with_pydub(in_wav: str, out_wav: str, silence_thresh: int = -40, min_silence_len: int = 300) -> str:
    audio = AudioSegment.from_wav(in_wav)
    nonsilent = pydub_silence.detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
    )
    if nonsilent:
        start, end = nonsilent[0][0], nonsilent[-1][1]
        trimmed = audio[start:end]
    else:
        trimmed = audio
    trimmed.export(out_wav, format="wav")
    return out_wav


RATE = 16000
DATA_ROOT = os.environ.get("DATA_ROOT", "enrolled")
SESSIONS_DIR = os.path.join(DATA_ROOT, "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

app = FastAPI(title="Speaker Auth WS API")


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    meta_path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if not os.path.exists(meta_path):
        return JSONResponse({"status": "not_found"}, status_code=404)
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/recorder", response_class=HTMLResponse)
def recorder(
    session_id: Optional[str] = None,
    target_path: Optional[str] = None,
    language: str = "vi-VN",
    auto: int = 0,
    max_ms: int = 12000,
    silence_ms: int = 1000,
    energy_thresh: float = 0.015,
    ws_base: Optional[str] = None,
):
    sid = session_id or uuid.uuid4().hex[:10]
    path = target_path or f"ws_{sid}.wav"
    btn_display = 'none' if auto else 'inline-block'
    info_display = 'none' if auto else 'block'
    status_display = 'none' if auto else 'inline-block'
    html = """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Recorder</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 16px; }}
    .box {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; max-width: 640px; }}
    button {{ margin-right: 8px; }}
    #status {{ margin-left: 8px; color: #444; }}
  </style>
</head>
<body>
  <h3 style=\"display:__INFO_DISPLAY__\">Browser Recorder → WebSocket</h3>
  <div class=\"box\">
    <div style=\"display:__INFO_DISPLAY__\">Session: <code>__SID__</code></div>
    <div style=\"display:__INFO_DISPLAY__\">Target: <code>__PATH__</code></div>
    <div style=\"margin-top:8px\">
      <button id=\"btnStart\" style=\"display:__BTN_DISPLAY__\">Start</button>
      <button id=\"btnStop\" disabled style=\"display:__BTN_DISPLAY__\">Stop</button>
      <span id=\"status\" style=\"display:__STATUS_DISPLAY__\">idle</span>
    </div>
    <div style=\"margin-top:6px;\">
      <div style=\"height:8px;background:#eee;border-radius:4px;overflow:hidden;\">
        <div id=\"lvl\" style=\"height:8px;background:#2b8a3e;width:0%\"></div>
      </div>
    </div>
    <div style=\"margin-top:6px;color:#555;display:__INFO_DISPLAY__\">WS: <code id=\"wsurl\"></code></div>
    <div id=\"transcript\" style=\"margin-top:8px;display:__INFO_DISPLAY__\"></div>
  </div>
  <script>
    const RATE = 16000;
    const params = new URLSearchParams(window.location.search);
    const sessionId = params.get('session_id') || '__SID__';
    const targetPath = params.get('target_path') || '__PATH__';
    const language = params.get('language') || '__LANG__';
    let WS_URL = '__WS_BASE__';
    if (!WS_URL || WS_URL === 'AUTO') {
      const proto = (location.protocol === 'https:') ? 'wss' : 'ws';
      WS_URL = `${proto}://${location.host}`;
    }
    WS_URL = WS_URL + `/ws/audio?session_id=${encodeURIComponent(sessionId)}&target_path=${encodeURIComponent(targetPath)}&language=${encodeURIComponent(language)}`;
    const AUTO = __AUTO__;
    const MAX_MS = __MAX_MS__;
    const SIL_MS = __SIL_MS__;
    const THRESH = __THRESH__;

    let ws = null, running = false, processor = null, source = null, audioContext = null;
    let startedSpeech = false, silenceAccum = 0, startedAt = 0;

    function downsampleBuffer(buffer, sampleRate, outSampleRate) {{
      if (outSampleRate === sampleRate) return buffer;
      const sampleRateRatio = sampleRate / outSampleRate;
      const newLength = Math.round(buffer.length / sampleRateRatio);
      const result = new Float32Array(newLength);
      let offsetResult = 0, offsetBuffer = 0;
      while (offsetResult < result.length) {{
        const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
        let accum = 0, count = 0;
        for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {{
          accum += buffer[i]; count++;
        }}
        result[offsetResult] = accum / count; offsetResult++; offsetBuffer = nextOffsetBuffer;
      }}
      return result;
    }}

    function floatTo16BitPCM(float32Array) {{
      const output = new Int16Array(float32Array.length);
      for (let i = 0; i < float32Array.length; i++) {{
        let s = Math.max(-1, Math.min(1, float32Array[i]));
        output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }}
      return output.buffer;
    }}

    async function start() {{
      if (running) return;
      const statusEl = document.getElementById('status');
      statusEl.innerText = 'requesting mic...';
      let stream;
      try {{
        stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
      }} catch (e) {{
        statusEl.innerText = 'mic error: ' + (e && e.message ? e.message : e);
        return;
      }}
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      if (audioContext.state === 'suspended') try {{ await audioContext.resume(); }} catch(e) {{}}
      source = audioContext.createMediaStreamSource(stream);
      processor = audioContext.createScriptProcessor(4096, 1, 1);
      statusEl.innerText = 'connecting ws...';
      const wsurlEl = document.getElementById('wsurl');
      if (wsurlEl) wsurlEl.innerText = WS_URL;
      try { ws = new WebSocket(WS_URL); }
      catch (e) { statusEl.innerText = 'ws ctor error'; return; }
      ws.binaryType = 'arraybuffer';
      ws.onopen = () => {
        statusEl.innerText = 'recording...';
        // Start audio flow only after WS is open
        try { source.connect(processor); processor.connect(audioContext.destination); } catch(e){}
        running = true;
        startedAt = Date.now();
        document.getElementById('btnStart').disabled = true;
        document.getElementById('btnStop').disabled = false;
      };
      ws.onmessage = (ev) => {{
        try {{
          const msg = JSON.parse(ev.data);
          if (msg.type === 'final') {{
            document.getElementById('transcript').innerText = 'Final: ' + (msg.transcript || '(empty)');
            // Server finalized; close soon
            try {{ ws.close(); }} catch(e){{}}
          }}
        }} catch(_) {{}}
      }};
      ws.onerror = (ev) => {{ statusEl.innerText = 'ws error'; }};
      ws.onclose = (ev) => {{ statusEl.innerText = 'closed ' + (ev.code || '') + (ev.reason ? (' ' + ev.reason) : ''); }};
      // Pre-speech buffer (to include small lead-in but skip long initial silence)
      const PRE_MS = 300; // keep up to 300ms before detected speech
      let preBuf = [];
      let preDurMs = 0;

      processor.onaudioprocess = (e) => {{
        if (!running) return;
        const input = e.inputBuffer.getChannelData(0);
        const down = downsampleBuffer(input, audioContext.sampleRate, RATE);
        // level meter & simple VAD
        let sum = 0; for (let i=0;i<down.length;i++) sum += Math.abs(down[i]);
        const avg = (sum / down.length);
        const lvl = Math.min(100, Math.round(avg * 400));
        document.getElementById('lvl').style.width = lvl + '%';

        const pcm16 = floatTo16BitPCM(down);

        if (!startedSpeech) {{
          // Pre-speech: buffer chunks up to PRE_MS without sending
          preBuf.push(pcm16);
          preDurMs += (e.inputBuffer.duration * 1000);
          while (preDurMs > PRE_MS && preBuf.length > 0) {{
            // drop oldest chunk
            const first = preBuf.shift();
            // Estimate duration reduction by same block duration
            preDurMs -= (e.inputBuffer.duration * 1000);
          }}
          if (avg > THRESH) {{
            startedSpeech = true;
            silenceAccum = 0;
            // Flush pre-buffered chunks first
            for (const chunk of preBuf) {{
              try {{ if (ws && ws.readyState === WebSocket.OPEN) ws.send(chunk); }} catch(e) {{}}
            }}
            preBuf = [];
            // Send current chunk too
            try {{ if (ws && ws.readyState === WebSocket.OPEN) ws.send(pcm16); }} catch(e) {{}}
          }}
          // else keep waiting until speech starts
        }} else {{
          // Already started: send chunk then update VAD for stopping
          try {{ if (ws && ws.readyState === WebSocket.OPEN) ws.send(pcm16); }} catch(e) {{}}
          if (avg > THRESH) {{
            silenceAccum = 0;
          }} else {{
            silenceAccum += (e.inputBuffer.duration * 1000);
          }}
          if (startedAt && (Date.now() - startedAt) >= MAX_MS) {{
            stopSendEnd(true);
          }} else if (silenceAccum >= SIL_MS) {{
            stopSendEnd(true);
          }}
        }}
      }};
      // Do not connect nodes yet; wait for ws.onopen
    }}

    function stop() {{
      if (!running) return;
      running = false;
      stopSendEnd();
      try {{ if (processor) processor.disconnect(); }} catch(e) {{}}
      try {{ if (source) source.disconnect(); }} catch(e) {{}}
      document.getElementById('btnStart').disabled = false;
      document.getElementById('btnStop').disabled = true;
    }}

    function stopSendEnd(fromVad=false) {{
      if (!running) return;
      running = false;
      // Give a small delay to allow last PCM frame to flush before END
      setTimeout(() => {{
        try {{ if (ws && ws.readyState === WebSocket.OPEN) ws.send('END'); }} catch(e) {{}}
        // Close a bit later to allow server to send 'final'
        setTimeout(() => {{ try {{ if (ws) ws.close(); }} catch(e) {{}} }}, 4000);
      }}, fromVad ? 50 : 0);
    }}

      document.getElementById('btnStart').onclick = start;
      document.getElementById('btnStop').onclick = stop;
      if (AUTO) {{ start(); }}
  </script>
</body>
</html>
"""
    html = (
        html
        .replace('__SID__', sid)
        .replace('__PATH__', path)
        .replace('__LANG__', language)
        .replace('__BTN_DISPLAY__', btn_display)
        .replace('__INFO_DISPLAY__', info_display)
        .replace('__STATUS_DISPLAY__', status_display)
        .replace('__AUTO__', '1' if auto else '0')
        .replace('__MAX_MS__', str(max_ms))
        .replace('__SIL_MS__', str(silence_ms))
        .replace('__THRESH__', str(energy_thresh))
    )
    # Convert any escaped double braces to single braces for valid JS/CSS
    html = html.replace('{{', '{').replace('}}', '}')
    # If ws_base was provided from query/Streamlit, inject it; else use AUTO to pick origin
    html = html.replace('__WS_BASE__', ws_base if ws_base else 'AUTO')
    return HTMLResponse(content=html)


def _safe_target_path(target_path: str) -> str:
    # Prevent directory traversal; keep under DATA_ROOT
    target_path = os.path.normpath(target_path)
    if os.path.isabs(target_path):
        # make it relative to root
        target_path = os.path.relpath(target_path, start="/")
    data_root_abs = os.path.abspath(DATA_ROOT)
    full_path_abs = os.path.abspath(os.path.join(data_root_abs, target_path))
    if not full_path_abs.startswith(data_root_abs + os.sep) and full_path_abs != data_root_abs:
        raise ValueError("Invalid target path")
    os.makedirs(os.path.dirname(full_path_abs), exist_ok=True)
    return full_path_abs


def write_wav_int16(frames: list[np.ndarray], out_path: str) -> str:
    if not frames:
        raise ValueError("No audio frames to write")
    audio = np.concatenate(frames, axis=0).astype(np.int16)
    sf.write(out_path, audio, RATE, subtype="PCM_16")
    return out_path


def transcribe_sync(wav_path: str, language: str = "vi-VN") -> str:
    client = speech.SpeechClient()
    with open(wav_path, "rb") as f:
        content = f.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language,
        enable_automatic_punctuation=True,
    )
    response = client.recognize(config=config, audio=audio)
    parts = []
    for result in response.results:
        if result.alternatives:
            parts.append(result.alternatives[0].transcript)
    return " ".join(s.strip() for s in parts if s).strip()


@app.websocket("/ws/audio")
async def ws_audio(
    websocket: WebSocket,
    session_id: Optional[str] = Query(default=None),
    target_path: Optional[str] = Query(default=None, description="Target WAV path relative to data root"),
    language: str = Query(default="vi-VN"),
):
    await websocket.accept()
    sid = session_id or uuid.uuid4().hex[:10]
    safe_path = None
    if target_path:
        try:
            safe_path = _safe_target_path(target_path)
        except Exception:
            await websocket.send_json({"type": "error", "message": "invalid target_path"})
            await websocket.close(code=4000)
            return

    meta_path = os.path.join(SESSIONS_DIR, f"{sid}.json")
    frames: list[np.ndarray] = []
    # Realtime STT setup
    stt_q: pyqueue.Queue[Optional[bytes]] = pyqueue.Queue()
    ws_msg_q: asyncio.Queue[dict] = asyncio.Queue()
    stt_final_text: str = ""

    def request_generator():
        while True:
            data = stt_q.get()
            if data is None:
                break
            yield speech.StreamingRecognizeRequest(audio_content=data)

    def stt_worker():
        nonlocal stt_final_text
        try:
            client = speech.SpeechClient()
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=RATE,
                language_code=language,
                enable_automatic_punctuation=True,
            )
            streaming_config = speech.StreamingRecognitionConfig(
                config=config,
                interim_results=True,
                single_utterance=False,
            )
            responses = client.streaming_recognize(streaming_config, requests=request_generator())
            for response in responses:
                if not response.results:
                    continue
                result = response.results[0]
                if not result.alternatives:
                    continue
                text = result.alternatives[0].transcript.strip()
                if result.is_final:
                    stt_final_text = text
                    # queue final message
                    try:
                        ws_msg_q.put_nowait({"type": "stt_final", "transcript": text})
                    except Exception:
                        pass
                else:
                    try:
                        ws_msg_q.put_nowait({"type": "stt_interim", "transcript": text})
                    except Exception:
                        pass
        except Exception as e:
            try:
                ws_msg_q.put_nowait({"type": "warn", "message": f"stt_stream_failed: {e}"})
            except Exception:
                pass

    async def ws_sender():
        while True:
            msg = await ws_msg_q.get()
            if msg is None:
                break
            try:
                await websocket.send_json(msg)
            except Exception:
                break
    try:
        await websocket.send_json({"type": "ready", "session_id": sid, "rate": RATE})

        # Start realtime STT worker and sender
        sender_task = asyncio.create_task(ws_sender())
        stt_thread = threading.Thread(target=stt_worker, daemon=True)
        stt_thread.start()
        while True:
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                data = msg["bytes"]
                # Expect 16-bit PCM little-endian mono @ 16 kHz
                frames.append(np.frombuffer(data, dtype=np.int16))
                # feed realtime STT
                try:
                    stt_q.put_nowait(data)
                except Exception:
                    pass
            elif "text" in msg and msg["text"] is not None:
                text = msg["text"].strip()
                if text.upper() == "END":
                    # end STT stream as well
                    try:
                        stt_q.put_nowait(None)
                    except Exception:
                        pass
                    break
                # Allow small control messages
                try:
                    payload = json.loads(text)
                    if payload.get("event") == "end":
                        try:
                            stt_q.put_nowait(None)
                        except Exception:
                            pass
                        break
                except Exception:
                    # Ignore non-JSON control text
                    pass
            else:
                # No payload — likely disconnect
                break

        if not frames:
            await websocket.send_json({"type": "error", "message": "no audio"})
            await websocket.close(code=4001)
            return

        out_wav = safe_path or os.path.join(DATA_ROOT, f"ws_{sid}.wav")
        write_wav_int16(frames, out_wav)
        # Trim trailing silence for a cleaner file
        try:
            trim_with_pydub(out_wav, out_wav, silence_thresh=-40, min_silence_len=400)
        except Exception:
            pass

        # Wait a short moment for the streaming worker to finish
        try:
            stt_thread.join(timeout=2.0)
        except Exception:
            pass
        transcript = stt_final_text
        if not transcript:
            # fallback to sync if streaming didn't produce a final
            try:
                transcript = transcribe_sync(out_wav, language=language)
            except Exception as e:
                await websocket.send_json({"type": "warn", "message": f"stt_failed: {e}"})

        # Persist metadata for Streamlit to pick up
        meta = {
            "session_id": sid,
            "wav_path": out_wav,
            "language": language,
            "transcript": transcript,
            "created_at": datetime.utcnow().isoformat(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        await websocket.send_json({"type": "final", "session_id": sid, "wav_path": out_wav, "transcript": transcript})
        # stop sender task
        try:
            await ws_msg_q.put(None)  # type: ignore
            await sender_task
        except Exception:
            pass
        await websocket.close(code=1000)
    except WebSocketDisconnect as e:
        # Client closed; finalize with what we have
        if frames:
            out_wav = safe_path or os.path.join(DATA_ROOT, f"ws_{sid}.wav")
            transcript = ""
            try:
                write_wav_int16(frames, out_wav)
                try:
                    trim_with_pydub(out_wav, out_wav, silence_thresh=-40, min_silence_len=400)
                except Exception:
                    pass
                # Prefer streaming final if available, else fallback
                try:
                    # signal streaming to stop if still running
                    try:
                        stt_q.put_nowait(None)
                    except Exception:
                        pass
                    # small wait for worker
                    _time.sleep(0.5)
                    transcript = stt_final_text or transcribe_sync(out_wav, language=language)
                except Exception:
                    transcript = stt_final_text or ""
            except Exception:
                pass
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "session_id": sid,
                    "wav_path": out_wav,
                    "language": language,
                    "transcript": transcript,
                    "aborted": False,
                    "code": getattr(e, 'code', None),
                }, f, ensure_ascii=False, indent=2)
        else:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"session_id": sid, "aborted": True, "code": getattr(e, 'code', None)}, f)
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close(code=1011)
