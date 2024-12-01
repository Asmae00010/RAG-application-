import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import tempfile
from PyPDF2 import PdfReader
import docx
import requests
import os
from datetime import datetime

# Initialize persistent directory for ChromaDB
PERSIST_DIRECTORY = "chroma_db"

# Custom prompt template
CUSTOM_PROMPT = """
Answer the following question based on the provided context. If the context doesn't contain the relevant information, 
explicitly mention that and then provide an answer based on your general knowledge.

Context: {context}

Question: {question}

Previous conversation:
{chat_history}

Answer:"""

# Theme handling
if 'theme' not in st.session_state:
    st.session_state.theme = "light"

if st.session_state.theme == "dark":
    st.markdown("""
        <style>
        .stApp {
            background-color: #262626 !important;
            color: #ffffff !important;
        }
        .stTextInput input, .stSelectbox select, .stSlider, .stButton>button {
            background-color: #404040 !important;
            color: #ffffff !important;
        }
        .stMarkdown {
            color: #ffffff !important;
        }
        .stSidebar {
            background-color: #1E1E1E !important;
        }
        .stExpander {
            background-color: #333333 !important;
        }
        </style>
    """, unsafe_allow_html=True)

def get_available_models():
    try:
        response = requests.get('http://localhost:11434/api/tags')
        models = [model['name'] for model in response.json()['models']]
        return models if models else ['llama2', 'mistral', 'codellama', 'aya-expanse']
    except:
        return ['llama2', 'mistral', 'codellama', 'aya-expanse']

def check_model_availability(model_name):
    try:
        response = requests.get(f'http://localhost:11434/api/show/{model_name}')
        return response.status_code == 200
    except:
        return False


def get_file_size_mb(file):
    file.seek(0, os.SEEK_END)
    size_mb = file.tell() / (1024 * 1024)
    file.seek(0)
    return size_mb

def process_file(file):
    MAX_FILE_SIZE_MB = 50
    
    file_size = get_file_size_mb(file)
    if file_size > MAX_FILE_SIZE_MB:
        st.error(f"File size ({file_size:.1f}MB) exceeds maximum limit of {MAX_FILE_SIZE_MB}MB")
        return None
        
    try:
        if file.type == "application/pdf":
            pdf = PdfReader(file)
            return " ".join(page.extract_text() for page in pdf.pages)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(file)
            return " ".join(para.text for para in doc.paragraphs)
        else:
            return file.getvalue().decode()
    except Exception as e:
        st.error(f"Error processing file {file.name}: {str(e)}")
        return None

def format_message(role, content, model_name):
    timestamp = datetime.now().strftime("%H:%M")
    if role == "user":
        return f"🧑 **You** ({timestamp}): {content}"
    return f"🤖 **{model_name}** ({timestamp}): {content}"

# Initialize session states
if 'conversation_histories' not in st.session_state:
    st.session_state.conversation_histories = {}
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'current_model' not in st.session_state:
    st.session_state.current_model = None

st.title("Multi-Model RAG Chat Application")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    theme = st.toggle("Dark Theme", value=st.session_state.theme == "dark")
    st.session_state.theme = "dark" if theme else "light"
    
    model_name = st.selectbox("Select Model", get_available_models())
    if not check_model_availability(model_name):
        st.warning(f"⚠️ {model_name} is not currently loaded in Ollama. Please pull it first.")
    
    # Initialize conversation history for new models
    if model_name not in st.session_state.conversation_histories:
        st.session_state.conversation_histories[model_name] = []
    
    # Advanced settings expander
    with st.expander("Advanced Settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
        top_k = st.slider("Top K Documents", 1, 5, 3)
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200)
    
    # Add a clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.conversation_histories[model_name] = []
    
    # Add a clear database button
    if st.button("Clear Document Database"):
        if os.path.exists(PERSIST_DIRECTORY):
            import shutil
            shutil.rmtree(PERSIST_DIRECTORY)
        st.session_state.vector_store = None
        st.success("Document database cleared!")

    # Document Statistics
    if st.session_state.vector_store is not None:
        st.markdown("### Document Statistics")
        doc_count = len(st.session_state.vector_store.get())
        st.markdown(f"Total Chunks: {doc_count}")

# Document upload section
uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing documents..."):
        all_text = ""
        progress_bar = st.progress(0)
        for i, file in enumerate(uploaded_files):
            progress_text = st.empty()
            progress_text.text(f"Processing {file.name}...")
            
            text = process_file(file)
            if text:
                all_text += text
                
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            
        progress_bar.empty()
        progress_text.empty()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        texts = text_splitter.split_text(all_text)
        
        try:
            embeddings = OllamaEmbeddings(model=model_name)
            
            # Initialize or update ChromaDB
            st.session_state.vector_store = Chroma.from_texts(
                texts,
                embeddings,
                persist_directory=PERSIST_DIRECTORY
            )
            st.success("Documents processed and indexed!")
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

# Chat interface
if st.session_state.vector_store is not None:
    try:
        # Initialize LLM and QA chain
        llm = Ollama(model=model_name, temperature=temperature)
        
        # Create custom prompt template
        prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=CUSTOM_PROMPT
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vector_store.as_retriever(
                search_kwargs={"k": top_k}
            ),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        
        # Display conversation history
        st.subheader(f"Conversation with {model_name}")
        if model_name in st.session_state.conversation_histories:
            for q, a in st.session_state.conversation_histories[model_name]:
                st.markdown(format_message("user", q, model_name))
                st.markdown(format_message("assistant", a, model_name))
                st.markdown("---")
        
        # Question input with columns
        col1, col2 = st.columns([4, 1])
        with col1:
            question = st.text_input("Ask a question:", key="question_input")
        with col2:
            send_button = st.button("Send", type="primary")
            
        if send_button:
            if not question:
                st.warning("Please enter a question!")
                st.stop()
                
            with st.spinner(f"Getting response from {model_name}..."):
                response = qa_chain({
                    "question": question,
                    "chat_history": st.session_state.conversation_histories[model_name]
                })
                
                st.session_state.conversation_histories[model_name].append(
                    (question, response["answer"])
                )
                
                # Display the latest response
                st.markdown(format_message("user", question, model_name))
                st.markdown(format_message("assistant", response["answer"], model_name))
                
                # Display source documents (optional)
                with st.expander("View Source Documents"):
                    for doc in response["source_documents"]:
                        st.write(doc.page_content)
                        st.write("---")
    except Exception as e:
        st.error(f"Error in chat interface: {str(e)}")
else:
    st.info("Please upload documents to start chatting.")



# Add a footer with information about the available models
st.markdown("---")
st.markdown("### Available Models")
for model in get_available_models():
    st.markdown(f"- {model}")