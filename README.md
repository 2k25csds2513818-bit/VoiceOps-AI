# VoiceOps AI — ZentroMart Customer Support Agent

Capstone for **From Tokens to Transformers: A Course on GenAI x LLM Mastery**.

## Features
- LLM-backed support answers with a grounded system prompt.
- Local retrieval over the supplied Markdown knowledge base; superseded policies are excluded.
- Conversation history for follow-up questions.
- Supplied mock `lookup_order` and `create_ticket` tools.
- Order identity check before details are shown.
- Gnani STT/TTS batch runner that saves transcript, reply text/audio, and retrieved source IDs.
- Streamlit chat interface with expandable retrieved-evidence panel.

## 1. Put the supplied dataset in place
Extract `Capstone_Project_Dataset_FTTT_2026.zip` beside this README. The folder must be:
`./Capstone_Project_Dataset_FTTT_2026/`
Do not commit a dataset copy if the organizers expect it to be supplied separately. The supplied `tools/` package must remain inside that folder.

## 2. Install
Python 3.10+ recommended.
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Validate the provided data:
```bash
python Capstone_Project_Dataset_FTTT_2026/validate_dataset.py
```

## 3. Configure environment
```bash
copy .env.example .env
```
On macOS/Linux use `cp .env.example .env`. Add your own LLM provider key to `OPENAI_API_KEY`. The code uses the OpenAI Python client and can be pointed at an OpenAI-compatible provider with `OPENAI_BASE_URL`. For voice, add your personal `GNANI_API_KEY`. Never commit `.env` or keys.

## 4. Run the chat app
```bash
streamlit run app.py
```
If no LLM key is configured, retrieval still works and the app clearly indicates that a key is needed for generated answers.

## 5. Run voice batch mode
Place your own consented test recordings in `audio_input/` (do not use real customer recordings). Then:
```bash
python run.py audio_input
```
The command transcribes audio using Gnani STT, passes the transcript to the same agent, synthesizes a spoken reply with Gnani TTS, and writes outputs under `outputs/`: `.txt`, `_reply.wav`, and `manifest.json`. Supported input extensions: wav, mp3, m4a, flac, ogg. Confirm the exact SDK call shape against your active Gnani account/docs if their SDK version changes.

## Example flow
1. Customer: "My order 89782837 hasn't arrived. Can I cancel it?"
2. Agent asks for registered phone's last four digits OR delivery pincode.
3. Agent reveals order details only after verification.
4. Customer asks a policy question; the agent retrieves relevant document sections.
5. If a human/exception is requested, the agent attempts to create a ticket and confirms it only if the tool returns `created=True`.

## Architecture
Audio → Gnani STT → conversation agent → TF-IDF document retrieval + LLM → Gnani TTS.
Order facts → supplied identity-checking mock tool.
Escalations → supplied ticket creation mock tool.

## Evaluation notes
Use `practice/questions.csv` and `practice/conversations.csv` to compare a prompt-only baseline with RAG and measure grounded accuracy, retrieval evidence, follow-up handling, handovers, and voice completion. Do not claim a score until you have run and recorded the tests. Add results and limitations to `report.pdf`.

## Known limitations
- TF-IDF is a lightweight lexical retrieval baseline, not a semantic embedding model; a stronger embedding/vector index is a good next improvement.
- The order verification helper is a synthetic educational simulation, not production authentication.
- Human escalation is simulated by writing a local CSV ticket.
- Voice requires a valid Gnani key, working network, compatible SDK and your own test audio.
- LLM answers require a valid API key and may need provider-specific model/base URL configuration.
