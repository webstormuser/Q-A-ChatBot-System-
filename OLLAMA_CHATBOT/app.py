import streamlit as st 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
import os 
from dotenv import load_dotenv
load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot system using Ollama"

# Define the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", "{question}"),
])

# Function to generate response
def generate_response(question, model_name, temperature, max_tokens):
    # Initialize the ChatGroq instance with API key and other parameters
    llm = ChatOllama(model=model_name, temperature=temperature, max_tokens=max_tokens)
    output = StrOutputParser()
    chain = prompt | llm | output
    answer = chain.invoke({'question': question})

    return answer

# Create Streamlit app
# Title of app
st.title("Enhanced Q&A Chatbot System with OLLAMA and LangChain")

st.sidebar.title("Settings")

# Drop down list to select various Groq API models
model_name = st.sidebar.selectbox("Select a Ollama model", ["Gemma 2"])
# Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input,model_name, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide a query")

try: 
    response = generate_response(user_input,model_name, temperature, max_tokens) 
    st.write(response) 
except Exception as e: 
    st.write(f"Error:{e}")