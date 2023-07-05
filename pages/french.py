#!/usr/bin/env python3

import streamlit as st

from llm_model import create_qa, translate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

TRANSFORMER_FR_EN = "Helsinki-NLP/opus-mt-fr-en"
TRANSFORMER_EN_FR = "Helsinki-NLP/opus-mt-en-fr"

def main():
    qa = create_qa() 

    with st.sidebar:
        "[![Ouverez le repo dans Github](https://github.com/codespaces/badge.svg)](https://github.com/AlleyCorpNord/privateChatbotGPT)"
    
    # Interactive questions and answers
    st.title("💬 Chatbot privé")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Comment puis-je vous aider ?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():

        english_prompt = translate(prompt, TRANSFORMER_FR_EN)

        st.session_state.messages.append({"role": "user", "content": english_prompt})
        st.chat_message("user").write(prompt)

        # Get the answer from the chain
        last_message = st.session_state.messages[-1]

        with st.spinner(text="En cours..."):
          res = qa(last_message["content"])

        # translate here
        answer = translate(res['result'], TRANSFORMER_EN_FR)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

if __name__ == "__main__":
    main()
