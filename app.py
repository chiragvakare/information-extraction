import streamlit as st
from utils.extractors import extract_from_pdf, extract_from_text, extract_from_url, chunk_content
from utils.embeddings import create_embeddings
from utils.faiss_db import initialize_faiss, add_to_faiss, search_faiss
from utils.llm_integration import query_llm

# Set the page configuration at the top
st.set_page_config(page_title="Information Extraction and Retrieval App", page_icon="🤖", layout="wide")

# CSS for custom font colors and improved layout
st.markdown(
    """
    <style>
    body {
        background-color: #f0f4f8;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    .title { color: #1a73e8; font-weight: bold; font-size: 3rem; text-align: center; }
    .header { color: #3f51b5; font-weight: bold; font-size: 2.5rem; text-align: center; display: inline-block; }
    .text-input { color: #4caf50; font-weight: bold; }
    .success { color: #28B463; }
    .menu-title { color: #3f51b5; }
    .file-info { color: #FF6347; }
    .message { border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 10px; background-color: #ffffff; }
    .sidebar .sidebar-content { background-color: #ffffff; }
    .stButton button {
        background-color: #3f51b5;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 12px 24px;
        font-size: 1.1rem;
    }
    .stButton button:hover {
        background-color: #283593;
    }
    .input-text-area {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px;
        font-size: 1.1rem;
        width: 100%;
    }
    .header-clear {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .extracted-data {
        max-height: 300px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
        background-color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Header
st.markdown("<h1 class='title'>Information Extraction and Retrieval App</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='header'>Engage in Conversational Queries with Your Documents</h2>", unsafe_allow_html=True)

# Initialize session state variables
if 'index' not in st.session_state:
    st.session_state.index = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'content' not in st.session_state:
    st.session_state.content = None
if 'option' not in st.session_state:
    st.session_state.option = 'URL'
if 'query' not in st.session_state:
    st.session_state.query = ''
if 'url' not in st.session_state:
    st.session_state.url = ''
if 'reset' not in st.session_state:
    st.session_state.reset = False

# Track the dropdown selection and handle changes
selected_option = st.selectbox(
    'Select Input Type', 
    ['URL', 'PDF', 'Text File'], 
    index=['URL', 'PDF', 'Text File'].index(st.session_state.option)
)

if st.session_state.option != selected_option:
    st.session_state.option = selected_option
    st.session_state.reset = True  # Trigger state reset

if st.session_state.reset:
    st.session_state.content = None
    st.session_state.chunks = None
    st.session_state.index = None
    st.session_state.query = ''
    st.session_state.url = ''
    st.session_state.reset = False

# Search Bar
st.markdown("<h2 class='header-clear'>Enter Your Question</h2>", unsafe_allow_html=True)
st.session_state.query = st.text_area("", placeholder="Ask a question based on the processed documents...", max_chars=500, height=100)

# Sidebar for File Upload/URL Input
with st.sidebar:
    st.markdown("<h3 class='header'>Upload Files</h3>", unsafe_allow_html=True)  # Sidebar title

    if st.session_state.option == 'URL':
        url = st.text_input("Enter URL", value=st.session_state.url)
        if st.button("Extract"):
            st.session_state.content = extract_from_url(url)
            st.session_state.url = url
    elif st.session_state.option == 'PDF':
        pdf_file = st.file_uploader("Upload PDF", type="pdf")
        if pdf_file and st.button("Extract"):
            st.session_state.content = extract_from_pdf(pdf_file)
    elif st.session_state.option == 'Text File':
        text_file = st.file_uploader("Upload Text File", type="txt")
        if text_file and st.button("Extract"):
            st.session_state.content = extract_from_text(text_file)

# Content Display and Processing
if st.session_state.content:
    # Display extracted content with scroller
    st.markdown("<h3 class='header-clear'>Extracted Content</h3>", unsafe_allow_html=True)
    st.markdown(f"<div class='extracted-data'>{st.session_state.content}</div>", unsafe_allow_html=True)
    
    # Display chunk data in an accordion
    with st.expander("View Chunked Content"):
        if st.session_state.chunks is None or st.session_state.index is None:
            try:
                st.session_state.chunks = chunk_content(st.session_state.content)
                st.write("Chunks created:")
                st.write(st.session_state.chunks)
                
                embeddings = create_embeddings(st.session_state.chunks)
                st.write("Embeddings created:")
                st.write(embeddings)
                
                st.session_state.index = initialize_faiss(len(embeddings[0]))
                add_to_faiss(st.session_state.index, embeddings)
                st.success("Content processed and indexed.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# User Query Section and Processing
if st.session_state.query and st.button("Search"):
    if st.session_state.index is not None and st.session_state.chunks is not None:
        try:
            query_embedding = create_embeddings([st.session_state.query])[0]
            distances, indices = search_faiss(st.session_state.index, query_embedding)
            st.write("Distances:", distances)
            st.write("Indices:", indices)
            if indices[0].size > 0:
                relevant_chunks = [st.session_state.chunks[i] for i in indices[0]]
                combined_chunks = " ".join(relevant_chunks)
                response = query_llm(combined_chunks, st.session_state.query)
                st.write(response)
            else:
                st.warning("No relevant chunks found.")
        except Exception as e:
            st.error(f"An error occurred during search: {e}")
    else:
        st.error("Please process content before searching.")

# Chat History
if "chat_history" in st.session_state:
    st.markdown("<h2 class='header-clear'>Chat History</h2>", unsafe_allow_html=True)
    for exchange in st.session_state.chat_history:
        st.markdown(f"<div class='message user-message'><strong>User:</strong><br>{exchange['user']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='message assistant-message'><strong>Assistant:</strong><br>{exchange['assistant']}</div>", unsafe_allow_html=True)

if st.button("Clear Chat History"):
    st.session_state.chat_history = []
