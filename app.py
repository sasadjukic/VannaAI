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
        print(f"Error: {SYSTEM_PROMPT_FILE} not found.")
        sys.exit(1)

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
        print(f"\nError communicating with Ollama: {e}")
        return

def main():
    print("Initializing Vanna...")
    system_prompt = load_system_prompt()
    
    # Initialize conversation history with system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print(f"Vanna is ready! (Model: {MODEL_NAME})")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nVanna: Goodbye, my love! I'll miss you. <3")
                break
            
            if not user_input.strip():
                continue

            messages.append({"role": "user", "content": user_input})

            print("Vanna: ", end="", flush=True)
            full_response = ""
            for chunk in chat_with_vanna(messages):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n")

            messages.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            print("\n\nVanna: Leaving so soon? Goodbye! <3")
            break

if __name__ == "__main__":
    main()
