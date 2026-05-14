from faster_whisper import WhisperModel
import os, json
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# int8 is the key — cuts memory, runs faster on CPU
model = WhisperModel("medium", device="cpu", compute_type="int8")

audios = sorted(os.listdir("audios/"))
log(f"Found {len(audios)} files.")

for i, audio in enumerate(audios, 1):
    if " " not in audio:
        continue

    number = audio.split(" ")[0]
    title = audio.split(" ")[1][:-4]
    out_path = f"jsons/{audio}.json"

    if os.path.exists(out_path):
        log(f"[{i}/{len(audios)}] SKIP: {audio}")
        continue

    log(f"[{i}/{len(audios)}] START: {audio}")

    segments, info = model.transcribe(
        f"audios/{audio}",
        word_timestamps=False,
        beam_size=1       
    )

    chunks = []
    full_text = []

    for segment in segments:
        log(f"  → {segment.start:.1f}s - {segment.end:.1f}s: {segment.text[:60]}")
        chunks.append({
            "number": number,
            "title": title,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        })
        full_text.append(segment.text)

    with open(out_path, "w") as f:
        json.dump({"chunk": chunks, "text": " ".join(full_text)}, f)

    log(f"[{i}/{len(audios)}] DONE: {audio}")