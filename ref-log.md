# Reference Log

## Libraries and Frameworks Used

- Streamlit - for the web interface
- LangChain - for text splitting (RecursiveCharacterTextSplitter) and document processing
- ChromaDB - vector database for storing document embeddings
- OpenAI API - using gpt-4o model and text-embedding-3-large for embeddings (through Cornell's API)
- PyPDF2 - for reading PDF files

## Course Resources

Started with the template provided in the assignment1 branch:
- chat_with_pdf.py - used as the main starting point
- langgraph_chroma_retreiver.ipynb - referenced for syntax examples
- requirements.txt and devcontainer config from the template

## Documentation References

- Streamlit docs (https://docs.streamlit.io/) - for chat interface and file upload
- LangChain docs (https://python.langchain.com/docs/) - for understanding text splitting and document loaders
- ChromaDB docs (https://docs.trychroma.com/) - for vector store setup
- PyPDF2 docs (https://pypdf2.readthedocs.io/) - for PDF text extraction

## AI Tools Used

Used Claude AI to help with:
- Understanding how to implement the RAG pipeline
- Debugging the numpy/pandas version conflicts
- Figuring out Streamlit session state for chat history
- Understanding ChromaDB and LangChain integration
- Code review and simplification

Main issues solved with AI help:
- Fixed numpy compatibility error by upgrading to numpy>=1.24.0
- Learned how to use RecursiveCharacterTextSplitter with appropriate chunk sizes
- Set up the vector search with k=10 retrievals

## Implementation Notes

All code was written by me after understanding the concepts. Used AI as a learning tool to understand the frameworks better, I asked for some code I cannot type out and imaging. Made sure to test everything and understand what each part does before including it in the final code.
