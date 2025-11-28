import streamlit as st
import requests
import json
import sys
import os

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gemma3:4b"
SYSTEM_PROMPT_FILE = "system_prompt.txt"

def load_system_prompt():
    """Loads the system prompt from the file."""
    try:
        with open(SYSTEM_PROMPT_FILE, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        st.error(f"Error: {SYSTEM_PROMPT_FILE} not found.")
        return None

def chat_with_vanna(messages):
    """Sends chat history to Ollama and yields the response."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                data = json.loads(decoded_line)
                if 'message' in data and 'content' in data['message']:
                    yield data['message']['content']
                if data.get('done', False):
                    break
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with Ollama: {e}")
        return

def main():
    st.set_page_config(page_title="Vanna AI", page_icon="💖")
    st.title("💖 Vanna AI Companion")

    # Load system prompt
    system_prompt = load_system_prompt()
    if not system_prompt:
        st.stop()

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": system_prompt}
        ]

    # Display chat messages (excluding system prompt)
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Say something to Vanna..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Stream response
            for chunk in chat_with_vanna(st.session_state.messages):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
        
        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
