import streamlit as st
import os
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.indexes.vectorstore import VectorStoreIndexWrapper

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.header("Chat with your PDF using Gemini & Astra DB")

# Sidebar for PDF Upload
with st.sidebar:
    st.title("Settings")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    process_button = st.button("Process PDF")

# Initialize Models
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.3)

# Logic for processing PDF
if uploaded_file and process_button:
    with st.spinner("Processing PDF..."):
        # Save temporary file
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and Split
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(docs)
        
        # Store in Astra DB
        vstore = AstraDBVectorStore(
            embedding=embeddings,
            collection_name="streamlit_pdf_demo",
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
            token=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        )
        vstore.add_documents(chunks)
        st.session_state.vector_index = VectorStoreIndexWrapper(vectorstore=vstore)
        st.success("PDF processed and stored!")

# Chat Interface
if "vector_index" in st.session_state:
    user_query = st.chat_input("Ask a question about your PDF...")
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        
        with st.spinner("Gemini is thinking..."):
            response = st.session_state.vector_index.query(user_query, llm=llm)
            with st.chat_message("assistant"):
                st.write(response)