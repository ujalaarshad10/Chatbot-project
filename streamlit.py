import streamlit as st
import torch
import os
from main import agent_executor
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

example_1="""
Eksempel 1:
input: Hvornår på ugen sender I pakker?
output: Pizza Fredag sender pakker ud hver dag, og hvis du bestiller inden 13:00, vil du få din ordre afsendt samme
dag. Vi leverer til hele landet.
Er der andet, jeg kan hjælpe med i dag?
"""
example_2="""
Eksempel 2:
input: Hvilken pizzaovn skal jeg vælge?
output: Når du skal vælge en pizzaovn, er det vigtigt at overveje, hvad du ønsker at bruge den til. Hvis du
er ny til pizzabagning, kan en Ooni Koda 16-pizzaovn være en god valg, da den er let at bruge og
kan give dig en god smagsoplevelse. Hvis du er mere erfaren og ønsker at lære mere om
pizzabagning, kan "Perfekt Pizza" af Jon Daniel Edlund være en god inspiration. Hvis du vil have
en ægte italiensk pizzasmag, kan Revolve Portable Pizzaovn være en god valg, da den er
certificeret af AVPN og kan give dig en autentisk Napolitansk pizzaoplevelse.
Er der noget andet, jeg kan hjælpe dig med?
"""


few_shot_prompt= f"""
{example_1}
{example_2}

"""

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
    recent_messages = st.session_state.messages[-3:-1]  # get the last 3 messages excluding current user input
    for message in recent_messages:
        if message["role"] == "user":
            chat_history.append({"type": "human", "content": message["content"]})
        elif message["role"] == "assistant":
            chat_history.append({"type": "ai", "content": message["content"]})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            input_prompt = f"{few_shot_prompt}\n Brug venligst de tilgængelige værktøjer (**search**, **retriever**) til at finde information. User Query: {prompt}"
            response = agent_executor.invoke({"input": input_prompt, "chat_history": chat_history})
            formatted_response = response
            st.markdown(formatted_response['output'])
            st.session_state.messages.append({"role": "assistant","content": formatted_response['output']})
            

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