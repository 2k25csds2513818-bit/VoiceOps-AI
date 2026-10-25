
from dotenv import load_dotenv
import os
from gnani.stt import GnaniSTTClient

load_dotenv(override=True)

print("API key loaded:", bool(os.getenv("GNANI_API_KEY")))

client = GnaniSTTClient()

result = client.transcribe(
    "test_reply.wav",
    language_code=os.getenv("GNANI_LANGUAGE", "en-IN")
)

print("STT result:", result)
