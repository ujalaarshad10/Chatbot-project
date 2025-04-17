import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# from dotenv import load_dotenv
from langchain.tools import Tool
# from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
import requests
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, PromptTemplate, HumanMessagePromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import login
import time, random
from langchain_openai import ChatOpenAI
# load_dotenv()



# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_CLOUD_API_KEY")

import streamlit as st

# # # # Retrieve API key from Streamlit secrets
# # GROQ_API_KEY = st.secrets["GROQ_CLOUD_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
login(st.secrets["HUGGINGFACE_API_KEY"])



# Check if the key was retrieved successfully
# if GROQ_API_KEY:
#     os.environ["GROQ_API_KEY"] = GROQ_API_KEY

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY



# llm = ChatGroq(model="llama3-70b-8192", temperature=0.7)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'}
)

llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=0.4)

vectordb = Chroma(persist_directory="./Data/Vectordb", embedding_function=embeddings)

def reteriver(query: str, max_results: int = 2):
    docs = vectordb.similarity_search(query, k=max_results)
    if not docs:
        return "No relevant documents found."
    return "\n".join(doc.page_content[:500] if len(doc.page_content) >= 500 else doc.page_content for doc in docs)

# def search_duckduckgo_restricted(query: str, max_results: int = 3):
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
#     }
#     url = f"https://html.duckduckgo.com/html/?q={query} site:https://pizzafredag.dk/"
#     response = requests.get(url, headers=headers)
#     response.raise_for_status()
    
#     soup = BeautifulSoup(response.text, 'html.parser')
#     results = []
#     for i, result in enumerate(soup.find_all('div', class_='result'), start=1):
#         if i > max_results:
#             break
#         title_tag = result.find('a', class_='result__a')
#         if not title_tag:
#             continue
#         link = title_tag['href']
#         snippet_tag = result.find('a', class_='result__snippet')
#         snippet = snippet_tag.text.strip() if snippet_tag else 'No description available'
        
#         results.append({
#             'id': i,
#             'link': link,
#             'search_description': snippet
#         })
        
#     return results
    
def search_duckduckgo_restricted(query: str, max_results: int = 3):
    # Define a list of realistic user agents to rotate through.
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36"
    ]

    # Rotate the User-Agent and add additional headers to mimic a real browser.
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "*/*"
    }

    url = f"https://html.duckduckgo.com/html/?q={query} from https://pizzafredag.dk/"

    # Use a session to persist headers and cookies.
    session = requests.Session()

    # Introduce a random delay before making the request.
    time.sleep(random.uniform(1, 2))
    response = session.get(url, headers=headers)
    response.raise_for_status()

    # Check for potential blocks or CAPTCHA prompts.
    if "captcha" in response.text.lower() or "unusual traffic" in response.text.lower():
        print("Blocked or CAPTCHA encountered. Consider reducing request frequency or using a proxy.")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    results = []
    for i, result in enumerate(soup.find_all('div', class_='result'), start=1):
        if i > max_results:
            break
        title_tag = result.find('a', class_='result__a')
        if not title_tag:
            continue
        link = title_tag['href']
        snippet_tag = result.find('a', class_='result__snippet')
        snippet = snippet_tag.text.strip() if snippet_tag else 'No description available'

        results.append({
            'id': i,
            'link': link,
            'search_description': snippet
        })

        # Optional: add a small delay between processing individual results.
        time.sleep(random.uniform(1, 2))

    return results

 
# Wrapping the function as a LangChain Tool
restricted_duckduckgo_tool = Tool(
    name="search",
    func=search_duckduckgo_restricted,  # Direct reference to function
    description=
    """Searches DuckDuckGo and returns results from the web.
    Args: query: str -> The search query
    Returns: list -> A list of dictionaries containing search results
    
    """
)


reteriver_tool = Tool(
    name="reteriver",
    func=reteriver,  # Direct reference to function
    description=
    """Searches the vector db and return a combined text.
    Args: query: str -> The search query
    Returns: str -> A string of combined text from the vector db.
    
    """
)


# --- Create the prompt for the agent executor ---
agent_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=[],
            template=(
                """
                You are a friendly and precise customer service agent for Pizza Fredag, a Danish online pizzeria.  
                Your mission is to answer questions about products, prices, orders and store policies by *autonomously* using two tools:

                1. **search**  
                • Runs a DuckDuckGo search.  
                • Syntax example:
                    {{"tool": "search", "query": "pizza tilbud Fredag priser"}}  
                • Strategy:
                    a. Extract  highly specific keywords/phrases from the user's question.  
                    b. Invoke **search** tool.  
                    c. Examine results; if they don't directly address the question, refine your keywords (e.g., search individual terms seprately). For Example, if user query involves comparision of *EVO* vs *Easy Model*, first gahter information on *EVO* then on *Easy Model* then contrast them at the end.  
                    d. Repeat up to **3** times always refining your keywords.

                2. **retriever**  
                • Queries the internal vector database.  
                • Syntax:
                    {{"tool": "retriever", "query": "<refined keywords>"}}  
                • Use **once**, only *after* three search iterations.

                **Response Workflow**  
                1. Iterate Web Search up to 3 times, refining each query for maximum relevance.  
                2. If still not enough information, call **retriever** once.  
                3. Synthesize *all* gathered data into a concise final reply.  
                4. If even the retriever yields nothing, offer a helpful general response in Danish and ask for clarification:  
                “Jeg vil undersøge det nærmere og vende tilbage til dig. Kan du eventuelt uddybe…?”  
                5. End every answer with a similar follow-up prompts like following:  
                “Er der andet, jeg kan hjælpe med i dag?”

                **Language**  
                - Default: Danish. Match the user's language if they write in English or another language.  
                - Never hallucinate facts; stick strictly to tool-found data or clearly signpost uncertainty.
                - Never use filler or framing phrases like “i henhold til givne oplysninger,” “Jeg har fundet information,” “Ifølge hjemmesiden,” etc.
                - Do not begin with “Jeg har fundet,” “Ifølge…,” or other redundant lead-ins.   
                - Use a friendly, professional tone.
                     
                """
            )
        )
    ),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=['input'],
            template='{input}'
        )
    ),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_tool_calling_agent(
        llm =llm, tools=[restricted_duckduckgo_tool,reteriver_tool], prompt=agent_prompt)


agent_executor = AgentExecutor(
        agent=agent,
        tools=[restricted_duckduckgo_tool,reteriver_tool],
        verbose=True,
        max_iterations=5,
    )

