
import streamlit as st
from src.agent import SupportAgent

st.set_page_config(
    page_title="VoiceOps AI | ZentroMart",
    page_icon="🎧",
    layout="wide"
)

st.title("🎧 VoiceOps AI")
st.caption("ZentroMart customer support · RAG + conversational AI")
st.info("Demo uses synthetic capstone data only. Order details require verification.")

WELCOME = {
    "role": "assistant",
    "content": (
        "Hi! I'm ZentroMart's AI support assistant. "
        "I can help with deliveries, returns, refunds and warranties. "
        "What can I help you with?"
    ),
    "sources": []
}

if "agent" not in st.session_state:
    st.session_state.agent = SupportAgent()

if "messages" not in st.session_state:
    st.session_state.messages = [WELCOME.copy()]

# Display the complete conversation and its evidence
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        sources = message.get("sources", [])
        if sources:
            with st.expander("Retrieved evidence"):
                for source in sources:
                    st.markdown(
                        f"**{source['id']}** · "
                        f"`{source['source']}` · "
                        f"relevance {source['score']:.3f}"
                    )
                    st.write(source["text"][:700])

prompt = st.chat_input("Ask about a policy or order…")

if prompt:
    # Save and display the user's message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "sources": []
    })

    with st.chat_message("user"):
        st.write(prompt)

    # Generate the response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = st.session_state.agent.respond(prompt)

        st.write(answer)

        if sources:
            with st.expander("Retrieved evidence"):
                for source in sources:
                    st.markdown(
                        f"**{source['id']}** · "
                        f"`{source['source']}` · "
                        f"relevance {source['score']:.3f}"
                    )
                    st.write(source["text"][:700])

    # Save both the response and its evidence
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

with st.sidebar:
    st.header("Project status")
    st.write("✅ Policy retrieval")
    st.write("✅ Conversation context")
    st.write("✅ Verified mock order tool")
    st.write("✅ Ticket handover")
    st.write("🎙️ Voice: run `python run.py audio_folder`")

    if st.button("Reset conversation"):
        st.session_state.agent = SupportAgent()
        st.session_state.messages = [WELCOME.copy()]
        st.rerun()
