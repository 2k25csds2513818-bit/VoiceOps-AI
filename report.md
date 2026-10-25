# VoiceOps AI — Capstone Report
**Course:** From Tokens to Transformers: A Course on GenAI x LLM Mastery  
**Organization brief:** Consulting & Analytics Club, IIT Guwahati × Gnani.ai  
**System:** ZentroMart AI customer-support agent

## 1. Problem and objective
ZentroMart receives repetitive questions about delivery, cancellation, returns, refunds and warranties. This project builds a conversational support agent that uses the supplied company knowledge base and synthetic order records. It supports text interaction and a batch voice path using Gnani speech-to-text (STT) and text-to-speech (TTS).

## 2. Implementation
**Step 1 — LLM and prompting.** The agent uses a system prompt that establishes the support role, requires the assistant to identify itself as AI, limits answers to evidence, and instructs it not to invent missing policies or claim tools succeeded without confirmation.

**Step 2 — Retrieval-augmented generation.** Markdown documents are split into section-level chunks and indexed with TF-IDF. The top matching chunks are provided to the LLM, and the interface records their section IDs and source filenames. Superseded documents are excluded from retrieval. TF-IDF is a transparent lexical baseline; semantic embeddings and a vector database are future improvements.

**Step 3 — Conversation and tools.** The agent retains recent turns, asks for an order ID and verifies the caller using either the registered phone suffix or delivery pincode before disclosing order details. It uses the provided mock order lookup and ticket creation functions. Escalation confirmation depends on the tool's `created=True` result.

**Step 4 — Voice.** The batch runner uses Gnani STT to transcribe input audio, passes text to the same agent, then uses Gnani TTS to create a spoken reply. It stores transcript, reply, source IDs and generated audio for reproducibility.

## 3. Evaluation
The dataset includes 52 single-question cases and 23 multi-turn conversations (96 customer turns). Run the provided practice questions and conversations, record results for a prompt-only baseline and the retrieval-enabled system, and report the measured counts here.

| Measure | Prompt-only baseline | RAG + tools |
|---|---:|---:|
| Correct, grounded answers | Not measured yet | Not measured yet |
| Retrieval evidence correct | N/A | Not measured yet |
| Conversation/tool decisions | Not measured yet | Not measured yet |
| Voice files processed end-to-end | N/A | Not measured yet |

**Important:** No performance scores are fabricated in this report. Replace “Not measured yet” after running evaluation.

## 4. Safety, privacy and limitations
Only the provided synthetic dataset should be used. Verification secrets must not be repeated in assistant replies, logs, or handover summaries. The mock verification tool is not production-grade authentication. API keys belong in `.env`, which must not be committed. Current limitations include lexical retrieval rather than semantic embeddings, provider-dependent LLM configuration, simulated ticket storage, and the need to test voice calls with an active Gnani account.

## 5. Future improvements
1. Add a semantic embedding model and persistent vector database.
2. Build an automated evaluation script for all practice questions and conversations.
3. Add stricter intent routing, date/business-day rules and explicit policy-based escalation.
4. Add a browser-based audio recorder and playback interface.
5. Improve multilingual handling and run adversarial tests for prompt injection and unsupported claims.

## 6. Reproducibility
Install `requirements.txt`, extract the supplied dataset beside the code, configure `.env` from `.env.example`, validate the dataset, run `streamlit run app.py`, and use `python run.py <audio_folder>` for voice batch evaluation.
