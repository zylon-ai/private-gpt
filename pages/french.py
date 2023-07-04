#!/usr/bin/env python3

import os
import argparse
import streamlit as st

from llm_model import create_qa
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


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
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        translated = model.generate(**tokenizer(prompt, return_tensors="pt", padding=True))
        tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    
        st.session_state.messages.append({"role": "user", "content": tgt_text[0]})
        st.chat_message("user").write(prompt)

        # Get the answer from the chain
        last_message = st.session_state.messages[-1]

        with st.spinner(text="En cours..."):
          res = qa(last_message["content"])

        # translate here
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        translated = model.generate(**tokenizer(res['result'], return_tensors="pt", padding=True))
        tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        answer = tgt_text[0]

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
