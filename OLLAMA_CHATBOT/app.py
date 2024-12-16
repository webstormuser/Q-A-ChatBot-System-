import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
import time
import httpx

load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot system using Ollama"

# Define the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot."),
    ("human", "{question}"),
])

# Function to generate response with retry mechanism
def generate_response(question, model_name, temperature, max_tokens):
    retries = 3
    for attempt in range(retries):
        try:
            llm = ChatOllama(model=model_name)
            output = StrOutputParser()
            chain = prompt | llm | output
            answer = chain.invoke({'question': question})
            return answer
        except httpx.ConnectError as e:
            if attempt < retries - 1:
                time.sleep(2)  # Wait for 2 seconds before retrying
                continue
            else:
                return f"Error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

# Create Streamlit app
st.title("Enhanced Q&A Chatbot System with OLLAMA and LangChain")

st.sidebar.title("Settings")

# Drop down list to select various Ollama models
model_name = st.sidebar.selectbox("Select an Ollama model", ["mistral"])

# Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, model_name, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide a query")

try:
    response = generate_response(user_input, model_name, temperature, max_tokens)
    st.write(response)
except Exception as e:
    st.write(f"Error: {e}")
