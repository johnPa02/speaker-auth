import streamlit as st
import os
import difflib
import random
import time
import json
from core import (
    SAMPLE_SENTENCES,
    verification,
)
from dotenv import load_dotenv
load_dotenv()

# ================= Streamlit Config =================
st.set_page_config(page_title="Speaker Authentication", page_icon="🎤", layout="wide")

if "enrollment_step" not in st.session_state:
    st.session_state.enrollment_step = 0
if "enrollment_sentences" not in st.session_state:
    st.session_state.enrollment_sentences = []
if "enrolled_files" not in st.session_state:
    st.session_state.enrolled_files = []
if "selected_user" not in st.session_state:
    st.session_state.selected_user = "user123"
if "current_transcript" not in st.session_state:
    st.session_state.current_transcript = ""
if "ws_session_id" not in st.session_state:
    st.session_state.ws_session_id = ""
if "ws_base" not in st.session_state:
    st.session_state.ws_base = os.environ.get("WS_BASE", "ws://localhost:8000")
if "verify_session_id" not in st.session_state:
    st.session_state.verify_session_id = ""
if "verify_started" not in st.session_state:
    st.session_state.verify_started = False


# ================= Helper Functions =================
def get_enrolled_users():
    users = []
    if os.path.exists("enrolled"):
        for file in os.listdir("enrolled"):
            if file.endswith("_enrolled_files.txt"):
                users.append(file.replace("_enrolled_files.txt", ""))
    return users


def get_user_enrollment_info(user_id):
    fpath = os.path.join("enrolled", f"{user_id}_enrolled_files.txt")
    if os.path.exists(fpath):
        with open(fpath) as f:
            files = [x.strip() for x in f.readlines() if x.strip()]
        return len(files), files
    return 0, []


def start_enrollment(user_id, num_enrollments=3):
    selected = random.sample(SAMPLE_SENTENCES, num_enrollments)
    st.session_state.enrollment_sentences = selected
    st.session_state.enrollment_step = 1
    st.session_state.enrolled_files = []
    os.makedirs("enrolled", exist_ok=True)


def render_ws_recorder(session_id: str, target_path: str, language: str = "vi-VN", ws_base: str = "ws://localhost:8000"):
    ws_path = f"/ws/audio?session_id={session_id}&target_path={target_path}&language={language}"
    html = '''
    <div style="padding:8px;border:1px solid #ddd;border-radius:8px;">
      <div style="font-weight:600;margin-bottom:6px;">Browser Recorder → WebSocket</div>
      <button id="btnStart">Start</button>
      <button id="btnStop" disabled>Stop</button>
      <span id="status" style="margin-left:8px;">idle</span>
      <div id="transcript" style="margin-top:8px;color:#333;"></div>
    </div>
    <script>
    const RATE = 16000;
    const WS_URL = '__WS_BASE__' + '__WS_PATH__';
    let ws = null;
    let running = false;
    let processor = null;
    let source = null;
    let audioContext = null;

    function downsampleBuffer(buffer, sampleRate, outSampleRate) {
      if (outSampleRate === sampleRate) {
        return buffer;
      }
      const sampleRateRatio = sampleRate / outSampleRate;
      const newLength = Math.round(buffer.length / sampleRateRatio);
      const result = new Float32Array(newLength);
      let offsetResult = 0;
      let offsetBuffer = 0;
      while (offsetResult < result.length) {
        const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
        let accum = 0, count = 0;
        for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
          accum += buffer[i];
          count++;
        }
        result[offsetResult] = accum / count;
        offsetResult++;
        offsetBuffer = nextOffsetBuffer;
      }
      return result;
    }

    function floatTo16BitPCM(float32Array){
      const output = new Int16Array(float32Array.length);
      for (let i = 0; i < float32Array.length; i++) {
        let s = Math.max(-1, Math.min(1, float32Array[i]));
        output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      return output.buffer;
    }

    async function start() {
      if (running) return;
      document.getElementById('status').innerText = 'requesting mic...';
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      if (audioContext.state === 'suspended') { try { await audioContext.resume(); } catch(_){} }
      source = audioContext.createMediaStreamSource(stream);
      processor = audioContext.createScriptProcessor(4096, 1, 1);
      ws = new WebSocket(WS_URL);
      ws.binaryType = 'arraybuffer';
      ws.onopen = () => { document.getElementById('status').innerText = 'recording...'; };
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'final') {
            document.getElementById('transcript').innerText = 'Final: ' + (msg.transcript || '(empty)');
          }
        } catch (_) {}
      };
      ws.onerror = (e) => { document.getElementById('status').innerText = 'ws error'; };
      ws.onclose = (ev) => { document.getElementById('status').innerText = 'closed ' + (ev.code || ''); };
      processor.onaudioprocess = (e) => {
        if (!running) return;
        const input = e.inputBuffer.getChannelData(0);
        const down = downsampleBuffer(input, audioContext.sampleRate, RATE);
        const pcm16 = floatTo16BitPCM(down);
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(pcm16);
      };
      source.connect(processor);
      processor.connect(audioContext.destination);
      running = true;
      document.getElementById('btnStart').disabled = true;
      document.getElementById('btnStop').disabled = false;
      // Keepalive
      const keepalive = setInterval(() => {
        try { if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({event:'ping', t: Date.now()})); } catch(_) {}
      }, 10000);
      ws.addEventListener('close', () => clearInterval(keepalive));
    }

    function stop() {
      if (!running) return;
      running = false;
      try { if (ws && ws.readyState === WebSocket.OPEN) ws.send('END'); } catch (e) {}
      try { if (ws) ws.close(); } catch (_) {}
      try { if (processor) processor.disconnect(); } catch (_) {}
      try { if (source) source.disconnect(); } catch (_) {}
      document.getElementById('btnStart').disabled = false;
      document.getElementById('btnStop').disabled = true;
    }

    document.getElementById('btnStart').onclick = start;
    document.getElementById('btnStop').onclick = stop;
    </script>
    '''
    html = html.replace('__WS_PATH__', ws_path).replace('__WS_BASE__', ws_base)
    st.components.v1.html(html, height=200)


def embed_iframe_recorder(session_id: str, target_rel: str, language: str = "vi-VN"):
    api_http_base = st.session_state.ws_base.replace('wss://', 'https://').replace('ws://', 'http://')
    rec_url = f"{api_http_base}/recorder?session_id={session_id}&target_path={target_rel}&language={language}&auto=1&ws_base={st.session_state.ws_base}"
    iframe_html = f"""
    <iframe src="{rec_url}" width="100%" height="80" style="border:0;" allow="microphone; autoplay"></iframe>
    """
    st.markdown(iframe_html, unsafe_allow_html=True)


def wait_for_session_meta(session_id: str, timeout_s: float = 20.0, poll_ms: int = 300):
    import time as _t
    start_t = _t.time()
    meta_path = os.path.join("enrolled", "sessions", f"{session_id}.json")
    while (_t.time() - start_t) < timeout_s:
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        _t.sleep(poll_ms / 1000.0)
    return None


def check_ws_session(session_id: str):
    meta_path = os.path.join("enrolled", "sessions", f"{session_id}.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ================= UI =================
st.markdown("<h1 style='text-align:center'>🎤 Speaker Authentication</h1>", unsafe_allow_html=True)

with st.sidebar:
    user_id = st.text_input("User ID", st.session_state.selected_user)
    if user_id != st.session_state.selected_user:
        st.session_state.selected_user = user_id
        st.session_state.enrollment_step = 0
    st.write("---")
    st.caption("WebSocket API base (ws://host:port)")
    ws_base = st.text_input("WS Base", st.session_state.ws_base)
    if ws_base and ws_base != st.session_state.ws_base:
        st.session_state.ws_base = ws_base
        st.rerun()

    st.write("### Enrolled users")
    for user in get_enrolled_users():
        num, _ = get_user_enrollment_info(user)
        if st.button(f"{user} ({num} samples)"):
            st.session_state.selected_user = user
            st.session_state.enrollment_step = 0
            st.rerun()

tab1, tab2 = st.tabs(["Enrollment", "Verification"])

# ================= Enrollment =================
with tab1:
    st.header(f"Enrollment: {st.session_state.selected_user}")
    num, _ = get_user_enrollment_info(st.session_state.selected_user)

    if st.session_state.enrollment_step == 0:
        st.info("Bấm Start Enrollment để bắt đầu")
        num_enrollments = st.slider("Number of samples", 1, 5, 3)
        if st.button("Start Enrollment"):
            start_enrollment(st.session_state.selected_user, num_enrollments)
            # reset ws session id for first step
            st.session_state.ws_session_id = ""
            st.rerun()

    elif st.session_state.enrollment_step <= len(st.session_state.enrollment_sentences):
        step = st.session_state.enrollment_step
        total = len(st.session_state.enrollment_sentences)
        sentence = st.session_state.enrollment_sentences[step - 1]

        st.write(f"Step {step}/{total}")
        st.success(f"📖 Please read: \"{sentence}\"")

        # Prepare session and target path under enrolled/
        if not st.session_state.ws_session_id:
            st.session_state.ws_session_id = f"enroll_{st.session_state.selected_user}_{step}_{int(time.time())}"
        session_id = st.session_state.ws_session_id
        target_rel = f"{st.session_state.selected_user}_enroll_{step}.wav"
        target_path = os.path.join("", target_rel)  # relative to DATA_ROOT on server

        st.caption("Recording starts automatically; it will stop after you speak and a brief pause.")
        embed_iframe_recorder(session_id, target_rel, language="vi-VN")
        with st.spinner("Recording and transcribing..."):
            meta = wait_for_session_meta(session_id, timeout_s=25.0, poll_ms=300)
        if not meta:
            st.error("Timed out waiting for recording. Click to retry.")
            if st.button("Retry this sentence"):
                st.session_state.ws_session_id = ""
                st.rerun()
        else:
            transcript = meta.get("transcript", "") if isinstance(meta, dict) else ""
            st.session_state.current_transcript = transcript
            st.write(f"📝 You said: {transcript}")
            sim = difflib.SequenceMatcher(None, transcript.lower(), sentence.lower()).ratio() if transcript else 0.0
            st.write(f"📊 Similarity: {sim:.2f}")
            saved_wav = os.path.join("enrolled", target_rel)
            if sim >= 0.8 and os.path.exists(saved_wav):
                st.success("✅ Passed this sentence!")
                st.session_state.enrolled_files.append(saved_wav)
                st.session_state.enrollment_step += 1
                st.session_state.ws_session_id = ""  # reset for next
                with st.spinner("Loading next sentence..."):
                    time.sleep(1.0)
                st.rerun()
            else:
                st.error("❌ Not correct or audio missing, please try again.")
                if st.button("Try again"):
                    st.session_state.ws_session_id = ""
                    st.rerun()

    else:
        st.success("🎉 Enrollment completed!")
        fpath = os.path.join("enrolled", f"{st.session_state.selected_user}_enrolled_files.txt")
        with open(fpath, "w") as f:
            for file in st.session_state.enrolled_files:
                f.write(file + "\n")

        if st.button("Start New Enrollment"):
            st.session_state.enrollment_step = 0
            st.session_state.current_transcript = ""
            st.session_state.ws_session_id = ""
            st.rerun()


# ================= Verification =================
with tab2:
    st.header("Verification")
    num, files = get_user_enrollment_info(st.session_state.selected_user)

    if num == 0:
        st.warning("User chưa được enroll.")
    else:
        st.info("Click Start to record; it will stop after a short pause.")
        if not st.session_state.verify_started:
            if st.button("Start Verification"):
                st.session_state.verify_session_id = f"verify_{st.session_state.selected_user}_{int(time.time())}"
                st.session_state.verify_started = True
                st.rerun()
        else:
            session_id = st.session_state.verify_session_id
            verify_rel = "verify.wav"  # under enrolled/
            embed_iframe_recorder(session_id, verify_rel, language="vi-VN")
            with st.spinner("Recording and preparing verification..."):
                meta = wait_for_session_meta(session_id, timeout_s=25.0, poll_ms=300)

            verify_wav = os.path.join("enrolled", verify_rel)
            if not os.path.exists(verify_wav):
                st.error("⚠️ No verification audio detected. Try again.")
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Retry Verification"):
                        st.session_state.verify_session_id = ""
                        st.session_state.verify_started = False
                        st.rerun()
            else:
                transcript = (meta.get("transcript", "") if isinstance(meta, dict) else "")
                st.session_state.current_transcript = transcript
                if transcript:
                    st.write(f"📝 Transcript: {transcript}")

                scores, preds = [], []
                for f in files:
                    score, pred = verification.verify_files(verify_wav, f)
                    score_val = float(score.item() if hasattr(score, "item") else score)
                    pred_val = bool(pred.item() if hasattr(pred, "item") else pred)
                    scores.append(score_val)
                    preds.append(pred_val)
                    st.write(f"Compare with {os.path.basename(f)}: {score_val:.3f} → {pred_val}")

                avg = sum(scores) / len(scores)
                majority = sum(preds) > len(preds) // 2
                st.write(f"Average score: {avg:.3f}")
                if avg >= 0.45 and majority:
                    st.success("✅ Verification successful!")
                else:
                    st.error("❌ Verification failed!")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("New Verification"):
                        st.session_state.verify_session_id = ""
                        st.session_state.verify_started = False
                        st.rerun()
