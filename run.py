"""Batch audio runner required by the capstone. Outputs transcript, reply audio and evidence."""
import sys, os, json, traceback
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)

def transcribe(path):
    from gnani.stt import GnaniSTTClient
    client = GnaniSTTClient()
    result = client.transcribe(str(path), language_code=os.getenv("GNANI_LANGUAGE", "en-IN"))
    return (result or {}).get("transcript", "").strip()

def speak(text, out_path):
    from gnani.tts import GnaniTTSClient
    client = GnaniTTSClient()
    client.synthesize(text, voice=os.getenv("GNANI_VOICE", "Kaveri"),
                      model=os.getenv("GNANI_TTS_MODEL", "timbre-v2.5"),
                      language=os.getenv("GNANI_TTS_LANGUAGE", "en-IN"),
                      output_file=str(out_path))

def main():
    if len(sys.argv) != 2:
        print("Usage: python run.py <folder_of_audio_files>")
        raise SystemExit(2)
    folder = Path(sys.argv[1])
    if not folder.is_dir():
        raise SystemExit(f"Audio folder not found: {folder}")
    from src.agent import SupportAgent
    agent = SupportAgent()
    out = Path("outputs"); out.mkdir(exist_ok=True)
    supported = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    audio_files = sorted(p for p in folder.iterdir() if p.suffix.lower() in supported)
    if not audio_files:
        raise SystemExit("No supported audio files found.")
    manifest = []
    for audio in audio_files:
        item = {"input": audio.name}
        try:
            transcript = transcribe(audio)
            item["transcript"] = transcript
            reply, sources = agent.respond(transcript)
            item["reply"] = reply
            item["retrieved_sources"] = [{"id":s["id"],"source":s["source"],"score":round(s["score"],4)} for s in sources]
            (out / f"{audio.stem}.txt").write_text(
                f"INPUT: {audio.name}\nTRANSCRIPT: {transcript}\nREPLY: {reply}\nSOURCES: " +
                ", ".join(s["id"] for s in sources), encoding="utf-8")
            speak(reply, out / f"{audio.stem}_reply.wav")
            item["status"] = "ok"
        except Exception as exc:
            item["status"] = "error"; item["error"] = str(exc)
            (out / f"{audio.stem}_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        manifest.append(item)
        print(f"{audio.name}: {item['status']}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved outputs to {out.resolve()}")

if __name__ == "__main__":
    main()
