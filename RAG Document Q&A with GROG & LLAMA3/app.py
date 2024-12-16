import streamlit as st 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq 
import os 
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

import os 
from dotenv import load_dotenv
load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot system using GROQ"
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

groq_api_key=os.getenv("GROQ_API_KEY")

llm = ChatGroq(api_key=groq_api_key,model="Gemma-7b-it")
