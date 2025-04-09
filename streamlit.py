import streamlit as st
import torch
import os
from main import agent_executor
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 


def init_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input(prompt: str):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Convert chat history to LangChain-compatible format
    chat_history = []
    for message in st.session_state.messages[:-1]:  # exclude current user input
        if message["role"] == "user":
            chat_history.append({"type": "human", "content": message["content"]})
        elif message["role"] == "assistant":
            chat_history.append({"type": "ai", "content": message["content"]})
        

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            input_prompt = f"Povide a precise response to following user query: {prompt}\n\n Search for relevant information on the website or vector database if search doesnot work.  Never mention useless things like from where you get information And give a useful response to the customer. If you dont have information then give a general response to help user or as the user to be specific."
            response = agent_executor.invoke({"input": input_prompt, "chat_history": chat_history})
            formatted_response = response
            st.markdown(f"**Input:** {prompt}\n\n**Output:** {formatted_response['output']}")
            st.session_state.messages.append({"role": "assistant", "content": f"**Input:** {prompt}\n\n**Output:** {formatted_response['output']}"})
            # st.markdown(f"{formatted_response}")

def main():
    # Set up Streamlit page
    st.set_page_config(page_title="PizzaFredag ChatBot", layout="wide")
    st.title("Pizza Fredag ChatBot")

    # Initialize chat history
    init_chat_history()

    # Display chat interface
    display_chat_history()

    # Chat input
    if prompt := st.chat_input("Ask any question..."):
        handle_user_input(prompt)

if __name__ == "__main__":
    main()