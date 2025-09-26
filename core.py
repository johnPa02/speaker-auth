import os
import sys
import queue
import difflib
import numpy as np
import sounddevice as sd
import soundfile as sf
from google.cloud import speech
from dotenv import load_dotenv
import random
from speechbrain.inference.speaker import SpeakerRecognition

load_dotenv()

SAMPLE_SENTENCES = [
    "Xin chào, tôi muốn đặt bàn cho hai người vào tối nay lúc bảy giờ.",
    "Bạn có thể kiểm tra giúp tôi còn bàn trống vào cuối tuần này không?",
    "Tôi muốn đặt bàn cạnh cửa sổ để có không gian yên tĩnh hơn.",
    "Xin vui lòng ghi chú rằng tôi bị dị ứng hải sản.",
    "Cho tôi hỏi nhà hàng có chỗ ngồi ngoài trời không?",
    "Tôi cần một bàn cho bốn người vào lúc tám giờ tối mai.",
    "Làm ơn xác nhận đặt "
    ""
    "chỗ của tôi qua số điện thoại này.",
    "Bạn có thể gửi tin nhắn xác nhận qua Zalo hoặc email không?",
    "Tôi muốn thay đổi thời gian đặt bàn sang chín giờ tối.",
    "Xin giữ cho tôi một bàn ở khu vực không hút thuốc.",
    "Tôi cần đặt bàn cho mười người vì có tiệc sinh nhật.",
    "Bạn có thể chuẩn bị thêm ghế em bé cho chúng tôi không?",
    "Tôi muốn đặt chỗ ở gần sân khấu để tiện theo dõi chương trình.",
    "Xin vui lòng giữ bàn cho tôi đến 7:30",
    "Bạn có thể giới thiệu một số món đặc biệt của nhà hàng không?",
    "Tôi muốn biết nhà hàng có chỗ đậu xe ô tô không.",
    "Xin hãy hủy đặt chỗ cũ và tạo đặt chỗ mới cho ngày mai.",
    "Tôi cần đặt bàn ăn trưa cho nhóm sáu người vào Chủ nhật.",
    "Bạn có thể cho tôi biết nhà hàng mở cửa đến mấy giờ không?",
    "Xin xác nhận lại tên và số điện thoại của tôi để hoàn tất đặt chỗ."
]

# ====== Load model ======
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

# ================= CONFIG =================
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
q = queue.Queue()

# ================= GOOGLE STREAMING STT =================
def request_generator():
    while True:
        data = q.get()
        if data is None:
            return
        yield speech.StreamingRecognizeRequest(audio_content=data)

from pydub import AudioSegment, silence

def trim_with_pydub(in_wav, out_wav, silence_thresh=-40, min_silence_len=300):
    """
    Trim silence ở cuối file bằng pydub.
    - silence_thresh: ngưỡng dBFS (mặc định -40 dBFS coi là im lặng)
    - min_silence_len: bao nhiêu ms im liên tục thì cắt
    """
    audio = AudioSegment.from_wav(in_wav)
    nonsilent = silence.detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )
    if nonsilent:
        start, end = nonsilent[0][0], nonsilent[-1][1]
        trimmed = audio[start:end]
    else:
        trimmed = audio
    trimmed.export(out_wav, format="wav")
    print(f"✂️ Trimmed silence (pydub) + saved: {out_wav}")
    return out_wav


def trim_leading_trailing_silence(audio_np, sr=16000, thresh=500,
                                  min_leading=2000, min_trailing=400):
    abs_audio = np.abs(audio_np)

    # Tìm điểm bắt đầu (leading)
    start = 0
    for i in range(len(abs_audio)):
        if abs_audio[i] > thresh:
            start = max(0, i - min_leading)
            break

    # Tìm điểm kết thúc (trailing)
    end = len(abs_audio)
    for i in range(len(abs_audio) - 1, 0, -1):
        if abs_audio[i] > thresh:
            end = min(len(abs_audio), i + min_trailing)
            break

    return audio_np[start:end]


def record_and_transcribe_with_audio(out_wav, language="vi-VN"):
    """Record from mic, send to Google STT, only save audio after speech detected (with pre-buffer), save WAV (trim silence)"""
    client = speech.SpeechClient()
    audio_data = []
    pre_buffer = []
    max_prebuffer_chunks = 10  # ~1s if CHUNK=1600, RATE=16000
    started_speech = False
    def mic_callback(indata, frames, time_, status):
        nonlocal started_speech
        if status:
            print(status, file=sys.stderr)
        data_bytes = bytes(indata)
        q.put(data_bytes)
        data_np = np.frombuffer(data_bytes, dtype=np.int16)
        if started_speech:
            audio_data.append(data_np)
        else:
            pre_buffer.append(data_np)
            if len(pre_buffer) > max_prebuffer_chunks:
                pre_buffer.pop(0)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language,
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
        single_utterance=False
    )
    transcript = ""
    with sd.RawInputStream(samplerate=RATE, blocksize=CHUNK, dtype="int16",
                           channels=1, callback=mic_callback):
        print("🎤 Please speak, the system will stop automatically when you are silent...")
        audio_requests = request_generator()
        responses = client.streaming_recognize(streaming_config, audio_requests)
        for response in responses:
            if not response.results:
                continue
            result = response.results[0]
            text = result.alternatives[0].transcript.strip()
            if not started_speech and text:
                started_speech = True
                print("🎙️ Speech detected → start saving audio...")
                if pre_buffer:
                    audio_data.extend(pre_buffer)
                    pre_buffer.clear()
            if result.is_final:
                transcript = text
                print(f"✅ Final transcript: {transcript}")
                break
            else:
                print(f"(interim) {text}")
    if audio_data:
        audio_np = np.concatenate(audio_data, axis=0)
        sf.write(out_wav, audio_np, RATE, subtype="PCM_16")
        out_wav = trim_with_pydub(out_wav, out_wav, silence_thresh=-40, min_silence_len=300)
        print(f"✂️ Trimmed silence + saved: {out_wav}")
    return transcript

def record_and_transcribe_multi(out_wav_prefix, num_utterances=3, language="vi-VN"):
    """
    Record multiple utterances in one session.
    - out_wav_prefix: prefix for saving audio files (e.g., "enrolled/user123_enroll")
    - num_utterances: how many utterances (sentences) to capture
    - language: language code for STT
    Returns:
        transcripts: list of transcripts (strings)
        wav_files: list of saved wav file paths
    """
    transcripts = []
    wav_files = []

    for idx in range(1, num_utterances + 1):
        out_wav = f"{out_wav_prefix}_{idx}.wav"
        print(f"\n🎤 Please speak sentence {idx}/{num_utterances} ...")

        transcript = record_and_transcribe_with_audio(out_wav, language=language)

        if not transcript.strip():
            print("⚠️ No speech detected for this utterance.")
            continue

        transcripts.append(transcript)
        wav_files.append(out_wav)

        print(f"✅ Final transcript {idx}: {transcript}")
        print(f"💾 Saved: {out_wav}")

    return transcripts, wav_files


# ================= ENROLLMENT =================
def enroll_user_google_streaming(user_id, sample_sentences, enrolled_dir="enrolled", num_enrollments=3):
    os.makedirs(enrolled_dir, exist_ok=True)
    print(f"\n🎯 Starting enrollment for {user_id}")
    enrolled_files = []
    if len(sample_sentences) < num_enrollments:
        raise ValueError("Not enough sample sentences to enroll.")
    selected_sentences = random.sample(sample_sentences, num_enrollments)
    for i, sentence in enumerate(selected_sentences):
        fname_wav = os.path.join(enrolled_dir, f"{user_id}_enroll_{i+1}.wav")
        while True:
            print(f"\n🔴 Enrollment {i+1}/{num_enrollments}")
            print(f"📖 Please read the following sentence:\n   \"{sentence}\"")
            transcript = record_and_transcribe_with_audio(fname_wav, language="vi-VN")
            print(f"📝 You just read: \"{transcript}\"")
            ratio = difflib.SequenceMatcher(None, transcript.lower(), sentence.lower()).ratio()
            print(f"📊 Transcript similarity: {ratio:.2f}")
            if ratio >= 0.8:
                print("✅ Valid sentence, moving to the next one.")
                enrolled_files.append(fname_wav)
                break
            else:
                print("❌ The sentence was not correct, please try again.")
    enrolled_files_path = os.path.join(enrolled_dir, f"{user_id}_enrolled_files.txt")
    with open(enrolled_files_path, 'w') as f:
        for file_path in enrolled_files:
            f.write(file_path + '\n')
    print(f"\n🎉 Enrollment completed with {num_enrollments} samples for {user_id}")
    return enrolled_files



# ================= VERIFICATION =================
def verify_user(user_id, enrolled_dir="enrolled"):
    """Verify against all enrolled audio files and calculate average score. Uses VAD-based recording and Google STT for transcript."""
    # Load enrolled files list
    enrolled_files_path = os.path.join(enrolled_dir, f"{user_id}_enrolled_files.txt")
    if not os.path.exists(enrolled_files_path):
        print(f"[ERROR] No enrollment found for user: {user_id}")
        return None, False
    with open(enrolled_files_path, 'r') as f:
        enrolled_files = [line.strip() for line in f.readlines() if line.strip()]
    if not enrolled_files:
        print(f"[ERROR] No enrolled audio files found for user: {user_id}")
        return None, False

    # Record verification audio (VAD-based, no fixed duration)
    verify_fname = "verify.wav"
    print(f"\n🔵 Verification - Please speak naturally...")
    transcript = record_and_transcribe_with_audio(verify_fname, language="vi-VN")
    print(f"📝 Transcript: '{transcript}'")

    # Calculate scores against all enrolled files
    scores = []
    predictions = []
    print("\n📊 Comparing with enrolled samples...")
    for i, enrolled_file in enumerate(enrolled_files):
        score, prediction = verification.verify_files(verify_fname, enrolled_file)
        score_val = float(score) if hasattr(score, '__float__') else score.item() if hasattr(score, 'item') else float(score)
        prediction_val = bool(prediction) if isinstance(prediction, (bool, int)) else bool(prediction.item())
        scores.append(score_val)
        predictions.append(prediction_val)
        print(f"Score with enrollment {i + 1}: {score_val:.4f} (prediction: {prediction_val})")

    # Calculate average score and decision
    avg_score = sum(scores) / len(scores)
    final_decision = avg_score > 0.45

    print(f"\n[VERIFY] Average score: {avg_score:.4f}")
    print(f"[VERIFY] Final decision: {final_decision}")

    if final_decision:
        print("✅ Verification successful!")
    else:
        print("❌ Verification failed!")

    # Clean up verify audio file
    if os.path.exists(verify_fname):
        os.remove(verify_fname)

    return avg_score, final_decision

# ================= DEMO =================
if __name__ == "__main__":

    # enroll_user_google_streaming("user123", SAMPLE_SENTENCES, num_enrollments=3)
    verify_user("user123")
