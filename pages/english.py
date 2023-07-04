#!/usr/bin/env python3
import streamlit as st

from llm_model import create_qa

def main():
    qa = create_qa()    
    with st.sidebar:
        "[![Open repo in GitHub](https://github.com/codespaces/badge.svg)](https://github.com/AlleyCorpNord/privateChatbotGPT)"
    
    # Interactive questions and answers
    st.title("💬 Private Chatbot")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Get the answer from the chain
        last_message = st.session_state.messages[-1]

        with st.spinner(text="In progress..."):
          res = qa(last_message["content"])
        
        answer = res['result']

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)


if __name__ == "__main__":
    main()
