import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# from dotenv import load_dotenv
from langchain.tools import Tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
import requests
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, PromptTemplate, HumanMessagePromptTemplate
# from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import login
import time, random


# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_CLOUD_API_KEY")
import streamlit as st

# # Retrieve API key from Streamlit secrets
GROQ_API_KEY = st.secrets["GROQ_CLOUD_API_KEY"]
login(st.secrets["HUGGINGFACE_API_KEY"])



# Check if the key was retrieved successfully
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY


llm = ChatGroq(model="llama-3.3-70b-versatile")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'}
)
vectordb = Chroma(persist_directory="./Data/Vectordb", embedding_function=embeddings)


def reteriver(query: str, max_results: int = 2):
    docs = vectordb.similarity_search(query, k=max_results)
    vector_info = "\n".join([doc.page_content for doc in docs])
    return vector_info

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
    
def search_duckduckgo_restricted(query: str, max_results: int = 2):
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
        "Accept": "text/html,application/xhtml+xml",
        "Referer": "https://duckduckgo.com/"
    }

    url = f"https://html.duckduckgo.com/html/?q={query} site:https://pizzafredag.dk/"

    # Use a session to persist headers and cookies.
    session = requests.Session()

    # Introduce a random delay before making the request.
    time.sleep(random.uniform(1, 3))
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
        time.sleep(random.uniform(0.5, 1.0))

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
                You are a helpful assistant at PizzaFredag. Your job is to answer user query in precise and accurate friendly mannner. 
                Use the given search tool to search there website for relevent information in Danish as the website is in Dansih, if needed and give a standard assistant response. 
                Reply in Danish only if the user query is in Danish. Always reply in the same language as the user query. Improve the search tool output by search the keywords first to get the understanding of what the product is.
                Just state the answer to the user to need to tell the user how you got the answer.
                Priotize Search over retriver tool, cause it is more accurate.
                if you dont get relevant information from the search tool, then use the retriever tool(important). 
                If one time search is not enough, use the search tool again with different queries.
                Analyse the given information with the user query and see if its relevant to the query. if its not then igonre it. And never mention your tools in response user is not interested in that. 
                Never mention how you got the information the customer is not intrusted in that.              
                Answwer the general conversation like "hi" or "hello" in gernal way without using tools.
                Don't provide false information if you don't have the information.
                Your reponse should only contain the answer to the user query. 
                Be specific to the query and give a precise answer. 
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
    max_iterations=4,
)

# # Callbacks support token-wise streaming
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# llm_final = ChatLlamaCpp(
#     model_path="Models/SnakModelQ4_K_M.gguf",
#     temperature=0.8,
#     max_tokens=500,
#     chat_format="llama-2",
#     top_p=0.9,
#     callback_manager=callback_manager,
#     verbose=True,  
# )





# # --- Define a function that builds the complete chain ---
# def process_query(query: str) -> str:
#     # 1. Get response from the agent executor (DuckDuckGo tool)
#     print("Running agent executor...")
#     input_query = f"Here is the user query: {query}\n\n Search for relevant information on the website and vector database. And give a useful response to the customer." 
#     agent_response = agent_executor.invoke({"input":input_query})
    
    
#     agent_response = agent_response['output']
#     print("\n\nRAW AGENT RESPONSE:",agent_response,"\n\n")
    
#     # 4. Create the final prompt template for ChatLlamaCpp
#     final_prompt_template = (
#         """
#         <s>[INST] <<SYS>>
#         Du er en hjælpsom assistent hos PizzaFreday. Din opgave er udelukkende at forbedre med hensyn til sproglig grammatik, 
#         ordforråd og relateret til brugerforespørgslen. Returner kun den forbedrede version uden yderligere forklaringer. 
#         Svar altid på dansk.
#         <</SYS>>
#         Forbedre sproget i følgende sammenhæng uden at ændre betydningen og relatere det til brugerforespørgslen         
#         Her er brugerforespørgsel: {query}
#         her er information: {context}
#         [/INST]
#         """
#     )
#     final_prompt = final_prompt_template.format(query=query,context=agent_response)
    
#     # 5. Get the final answer from ChatLlamaCpp
#     print("Generating final answer with ChatLlamaCpp...")
#     final_answer = llm_final.invoke(final_prompt)
#     return final_answer

