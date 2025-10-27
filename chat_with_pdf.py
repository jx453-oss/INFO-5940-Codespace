import streamlit as st
import os
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Initialize OpenAI client
client = OpenAI(
    api_key="sk-2gKXW6iepBgSh9wLTHr35w",
    base_url="https://api.ai.it.cornell.edu",
)

def extract_text_from_file(uploaded_file):
    """Extract text from uploaded .txt or .pdf file"""
    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension == 'txt':
        return uploaded_file.read().decode("utf-8")

    elif file_extension == 'pdf':
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text

    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def create_document_chunks(text, source_name):
    """Split text into chunks for better retrieval"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(text)

    # Create Document objects with metadata
    documents = [
        Document(
            page_content=chunk,
            metadata={"source": source_name, "chunk_id": i}
        )
        for i, chunk in enumerate(chunks)
    ]

    return documents

def initialize_vectorstore(documents):
    """Create vector database from document chunks"""
    embeddings = OpenAIEmbeddings(
        model="openai.text-embedding-3-large",
        openai_api_key="sk-2gKXW6iepBgSh9wLTHr35w",
        openai_api_base="https://api.ai.it.cornell.edu"
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="document_collection"
    )

    return vectorstore

def retrieve_relevant_chunks(vectorstore, question, k=10):
    """Retrieve most relevant chunks based on question"""
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))
    return context, sources

# Streamlit UI
st.title("File Q&A with RAG")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

if "uploaded_files_names" not in st.session_state:
    st.session_state["uploaded_files_names"] = []

# Sidebar for file upload and display
st.sidebar.header("Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload documents",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

# Process uploaded files
if uploaded_files:
    current_file_names = [f.name for f in uploaded_files]

    if current_file_names != st.session_state["uploaded_files_names"]:
        with st.sidebar.status("Processing documents...", expanded=True) as status:
            all_documents = []

            for uploaded_file in uploaded_files:
                st.write(f"Processing: {uploaded_file.name}")

                # Extract text and create chunks
                text = extract_text_from_file(uploaded_file)
                chunks = create_document_chunks(text, uploaded_file.name)
                all_documents.extend(chunks)

                st.write(f"Created {len(chunks)} chunks")

            # Create vector database
            st.write(f"Creating vector database with {len(all_documents)} chunks...")
            vectorstore = initialize_vectorstore(all_documents)

            # Save to session state
            st.session_state["vectorstore"] = vectorstore
            st.session_state["uploaded_files_names"] = current_file_names

            status.update(label="Documents processed successfully", state="complete")

        # Reset chat history
        st.session_state["messages"] = [
            {"role": "assistant", "content": f"Loaded {len(uploaded_files)} document(s). You can ask questions now!"}
        ]

# Display uploaded files
if st.session_state["uploaded_files_names"]:
    st.sidebar.success(f"Loaded {len(st.session_state['uploaded_files_names'])} document(s)")
    st.sidebar.write("**Uploaded files:**")
    for name in st.session_state["uploaded_files_names"]:
        st.sidebar.write(f"- {name}")

# Chat interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg:
            with st.expander("Sources"):
                for source in msg["sources"]:
                    st.write(f"- {source}")

# Chat input
question = st.chat_input(
    "Ask a question about the documents",
    disabled=st.session_state["vectorstore"] is None
)

# Handle user question
if question and st.session_state["vectorstore"]:
    # Display user question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # RAG retrieval and generation
    with st.chat_message("assistant"):
        with st.status("Thinking...", expanded=False) as status:
            # Retrieve relevant chunks
            st.write("Searching documents...")
            context, sources = retrieve_relevant_chunks(
                st.session_state["vectorstore"],
                question,
                k=10
            )

            st.write(f"Found relevant content from {len(sources)} document(s)")

            # Build RAG prompt
            system_prompt = f"""You are a helpful assistant. Answer the question based only on the following context.
If you don't know the answer, say "I don't have enough information to answer this question."
Keep your answer concise (max 3 sentences).

Context:
{context}
"""

            status.update(label="Generating answer...", state="running")

            # Call LLM
            stream = client.chat.completions.create(
                model="openai.gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                stream=True
            )

            status.update(label="Complete", state="complete")

        # Display answer
        response = st.write_stream(stream)

        # Display sources
        with st.expander("Sources"):
            for source in sources:
                st.write(f"- {source}")

    # Save to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources
    })

# Initial info message
if not st.session_state["vectorstore"]:
    st.info("Please upload documents using the sidebar to get started.")
