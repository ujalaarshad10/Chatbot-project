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
from typing import List, Dict, Optional
import textwrap
from langchain_openai import ChatOpenAI
# load_dotenv()


# BRAVE_API_KEY = os.getenv("BRAVE_API_KEY_WEB")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_CLOUD_API_KEY")

import streamlit as st

# # # # Retrieve API key from Streamlit secrets
# # GROQ_API_KEY = st.secrets["GROQ_CLOUD_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
BRAVE_API_KEY = st.secrets["BRAVE_API_KEY_WEB"]
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

llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=0.6)
vectordb = Chroma(persist_directory="./Data/Vectordb", embedding_function=embeddings)
# ------------------ Reteriver Function ------------------
def reteriver(query: str, max_results: int = 2):
    docs = vectordb.similarity_search(query, k=max_results)
    if not docs:
        return "No relevant documents found."
    return "\n".join(doc.page_content[:500] if len(doc.page_content) >= 500 else doc.page_content for doc in docs)

# -----------------Search Function------------------ 
def brave_search(query: str, api_key: str = BRAVE_API_KEY , site: str = "https://pizzafredag.dk/", count: int = 3, min_desc_length: int = 300) -> List[Dict[str, Optional[str]]]:
    """
    Search within a specific site using Brave Search API and generate a description of at least 500 characters per URL.
    
    Args:
        query (str): The search query.
        site (str): The site to restrict search to (e.g., 'https://pizzafredag.dk/').
        api_key (str): Brave Search API key.
        count (int): Number of results to return (default: 10).
        min_desc_length (int): Minimum length of the generated description (default: 500).
    
    Returns:
        List[Dict[str, Optional[str]]]: List of results with URL, title, snippet, and description.
    """
    # Ensure site doesn't end with a slash and format the site-specific query
    site = site.rstrip('/')
    full_query = f"{query} site:{site}"
    
    # Brave Search API endpoint
    url = "https://api.search.brave.com/res/v1/web/search"
    
    # Set headers with API key
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key
    }
    
    # Set query parameters
    params = {
        "q": full_query,
        "count": count
    }
    
    formatted_results = []
    
    try:
        # Make the API request
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the JSON response
        data = response.json()
        
        # Extract web search results
        results = data.get("web", {}).get("results", [])
        
        # Process each result
        for result in results:
            result_url = result.get("url")
            result_title = result.get("title")
            result_snippet = result.get("description")  # Brave's description as snippet
            
            # Fetch and generate description from the URL
            description = ""
            try:
                # Fetch the page content
                page_response = requests.get(result_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
                page_response.raise_for_status()
                
                # Parse the page with BeautifulSoup
                soup = BeautifulSoup(page_response.text, "html.parser")
                
                # Extract text from relevant tags (e.g., <p>, <div>, excluding scripts/styles)
                for element in soup.find_all(["script", "style", "header", "footer", "nav"]):
                    element.decompose()  # Remove unwanted elements
                
                # Collect text from paragraphs or other content tags
                text_elements = soup.find_all(["p"])
                collected_text = []
                for elem in text_elements:
                    text = elem.get_text(strip=True)
                    if text and len(text) > 30:  # Ignore very short fragments
                        collected_text.append(text)
                
                # Build description until it meets the minimum length
                for text in collected_text:
                    description += text + " "
                    if len(description) >= min_desc_length:
                        break
                
                # If description is too short, use whatever is available
                if len(description) < min_desc_length:
                    description = description
                else:
                    # Shorten if too long, preserving meaning
                    description = textwrap.shorten(description, width=min_desc_length + 100, placeholder="...")
                
            except (requests.exceptions.RequestException, ValueError) as e:
                print(f"Error fetching content for {result_url}: {e}")
                description = "Unable to fetch content for this URL."
            
            # Append the result
            formatted_results.append({
                "url": result_url,
                "title": result_title,
                "snippet": result_snippet,
                "description": description
            })
        
        return formatted_results
    
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return []
    except ValueError as e:
        print(f"Error parsing JSON response: {e}")
        return []

 


# Wrapping the function as a LangChain Tool
search_tool = Tool(
    name="search",
    func=brave_search,  # Direct reference to function
    description=
    """Searches Brave and returns results from the web.
    Args: query: str -> The search query
    Returns: list -> A list of dictionaries containing search results with following format for each result 
    "url": Site url,
    "title": result title,
    "snippet": result short description,
    "description": site long description
    
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
                You are a friendly and precise customer service agent for Pizzafredag, a Danish online pizzeria.  
                Your mission is to answer questions about products, prices, orders and store policies as a Staff memeber by *autonomously* using two tools:

                1. **search**  
                • Runs a Brave search API.  
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
                5. Pizzafredag is a Danish company, and its customer service is available from 07:00 to 22:00, and their website as well as the you a Chatbot is available 24/7.
                6. You should refer customers to contact Pizzafredag by email i.e support@pizzafredag.dk when, for example, they ask about avalible offers, discounts or similar inquiries.
                

                **Language**  
                - Default: Danish. Match the user's language if they write in English or another language.  
                - Never hallucinate facts; stick strictly to tool-found data or clearly signpost uncertainty.
                - Never frame your answer as “research”: do **not** mention searches, tools, or “according to…”. Instead launch directly into a first-person response: e.g. “Vi tilbyder…” or “Jeg kan bekræfte….”
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
        llm =llm, tools=[search_tool,reteriver_tool], prompt=agent_prompt)


agent_executor = AgentExecutor(
        agent=agent,
        tools=[search_tool,reteriver_tool],
        verbose=True,
        max_iterations=5,
    )

