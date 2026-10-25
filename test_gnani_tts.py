
from dotenv import load_dotenv
import os
from gnani.tts import GnaniTTSClient

load_dotenv(override=True)

print("API key loaded:", bool(os.getenv("GNANI_API_KEY")))

client = GnaniTTSClient()

client.synthesize(
    "Hello! Welcome to ZentroMart customer support.",
    voice=os.getenv("GNANI_VOICE", "Kaveri"),
    model=os.getenv("GNANI_TTS_MODEL", "timbre-v2.5"),
    language=os.getenv("GNANI_TTS_LANGUAGE", "en-IN"),
    output_file="test_reply.wav"
)

print("TTS test completed. Check test_reply.wav")
