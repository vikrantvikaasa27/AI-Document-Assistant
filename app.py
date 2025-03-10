import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
import dotenv
dotenv.load_dotenv()

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    .stChatInput input { background-color: #1E1E1E !important; color: #FFFFFF !important; border: 1px solid #3A3A3A !important; }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) { background-color: #1E1E1E !important; border: 1px solid #3A3A3A !important; color: #E0E0E0 !important; border-radius: 10px; padding: 15px; margin: 10px 0; }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) { background-color: #2A2A2A !important; border: 1px solid #404040 !important; color: #F0F0F0 !important; border-radius: 10px; padding: 15px; margin: 10px 0; }
    .stChatMessage .avatar { background-color: #00FFAA !important; color: #000000 !important; }
    .stChatMessage p, .stChatMessage div { color: #FFFFFF !important; }
    .stFileUploader { background-color: #1E1E1E; border: 1px solid #3A3A3A; border-radius: 5px; padding: 15px; }
    h1, h2, h3 { color: #00FFAA !important; }
    </style>
    """, unsafe_allow_html=True)

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

PDF_STORAGE_PATH = 'document_store/pdfs/'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
# NVIDIA Chat Model
NVIDIA_CHAT_MODEL = ChatNVIDIA(
    model="meta/llama-3.3-70b-instruct",
    api_key=NVIDIA_API_KEY,
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
)

def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted_prompt = conversation_prompt.format(user_query=user_query, document_context=context_text)

    response = ""
    for chunk in NVIDIA_CHAT_MODEL.stream([{"role": "user", "content": formatted_prompt}]):
        response += chunk.content
    return response

# UI Configuration
st.title(" 🤖 LangChain AI Doc Assistant")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False
)

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)
    
    st.success("✅ Document processed successfully! Ask your questions below.")
    
    user_input = st.chat_input("Enter your question about the document...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
        
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)
