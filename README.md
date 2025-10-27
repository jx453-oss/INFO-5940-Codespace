# INFO 5940 Assignment 1: RAG Q&A System

## Overview

This is a document question-answering system using Retrieval-Augmented Generation (RAG). Users can upload documents (txt or pdf) and ask questions about them. The system searches for relevant content in the documents and generates answers using GPT-4o.

## Requirements Completed

This project fulfills all assignment requirements (200 points total):

1. **Codespace Setup (10 points)** - Uses the provided template with requirements.txt
2. **File Upload (10 points)** - Supports txt file upload with a user interface
3. **RAG System (150 points)**:
   - Document chunking (50 points): RecursiveCharacterTextSplitter with 800-character chunks and 100-character overlap
   - RAG pipeline (50 points): ChromaDB vector database, OpenAI embeddings, similarity search with k=10
   - Chat interface (50 points): Streamlit chat UI with streaming responses and source display
4. **PDF Support (15 points)** - Extended to support both txt and pdf files using PyPDF2
5. **Multiple Documents (15 points)** - Can upload and search across multiple documents

## Installation and Running

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run chat_with_pdf.py
```

3. Open your browser to http://localhost:8501

## How to Use

1. Upload documents using the "Browse files" button in the left sidebar
2. Wait for the system to process the documents (you'll see progress messages)
3. Type your question in the chat input at the bottom
4. View the answer and click "Sources" to see which documents were used

## Testing

You can test with the provided sample file `data/RAG_source.txt`. Try questions like:
- "What is Zelomax?"
- "Is Zelomax safe during pregnancy?"
- "What is Xenthera used for?"

## System Architecture

The system works in these steps:

1. User uploads documents (txt or pdf files)
2. Text is extracted (directly for txt, using PyPDF2 for pdf)
3. Documents are split into 800-character chunks with 100-character overlap
4. Chunks are converted to vectors using OpenAI's text-embedding-3-large model
5. Vectors are stored in ChromaDB
6. When user asks a question, the system searches for the 10 most similar chunks
7. These chunks are used as context for GPT-4o to generate an answer
8. The answer is displayed with links to source documents

## Technical Stack

- **Frontend**: Streamlit
- **Document Processing**: PyPDF2 for PDFs, LangChain for text splitting
- **Embeddings**: OpenAI text-embedding-3-large
- **Vector Database**: ChromaDB
- **LLM**: GPT-4o (via Cornell API)

## Code Structure

Main file is `chat_with_pdf.py` with these key functions:

- `extract_text_from_file()` - Reads text from uploaded files
- `create_document_chunks()` - Splits text into chunks
- `initialize_vectorstore()` - Creates and populates the vector database
- `retrieve_relevant_chunks()` - Searches for relevant content

The UI has three main parts:
- Sidebar for file upload and showing loaded files
- Main chat area for conversation
- Bottom input box for questions

## Files

- `chat_with_pdf.py` - Main application code
- `requirements.txt` - Python dependencies
- `data/RAG_source.txt` - Sample test document
- `langgraph_chroma_retreiver.ipynb` - Reference notebook (not required for assignment)

## Notes

The system uses a simple RAG approach without LangGraph for better clarity. The chunk size of 800 with 100 overlap was chosen to balance context preservation and retrieval precision. The system stores vectors in memory during the session and doesn't persist them to disk.

## Troubleshooting

If the app doesn't start, make sure all dependencies are installed with `pip install -r requirements.txt`.

If PDFs can't be read, make sure they contain actual text (not scanned images).

For questions not being answered accurately, try being more specific or check the "Sources" section to see what content was retrieved.
