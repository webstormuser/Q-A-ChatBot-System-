import streamlit as st 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq  # Ensure you have the correct import

import os 
from dotenv import load_dotenv
load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

# Define the prompt
prompt = ChatPromptTemplate([
    ("system", "You are a helpful AI bot."),
    ("human", "{question}"),
])

# Function to generate response
def generate_response(question, api_key, model_name, temperature, max_tokens):
    # Initialize the ChatGroq instance with API key and other parameters
    llm = ChatGroq(api_key=api_key, model=model_name, temperature=temperature, max_tokens=max_tokens)
    output = StrOutputParser()
    chain = prompt | llm | output
    answer = chain.invoke({'question': question})

    return answer

# Create Streamlit app
# Title of app
st.title("Enhanced Q&A Chatbot System with GROQ API and LangChain")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your GROQ API Key", type="password")

# Drop down list to select various Groq API models
model_name = st.sidebar.selectbox("Select a GROQ model", ["gemma2-9b-it", "gemma2-7b-it"])

# Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, api_key, model_name, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide a query")
