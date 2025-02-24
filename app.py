import os

import streamlit as st
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# App title
st.title("PDF Question Answer System")

# File Uploader
upload_file = st.file_uploader("Upload a Pdf file", type = "pdf")

# initialize session_state for QA chain
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

def process_pdf(file):
    # Save the file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(file.getbuffer())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents=loader.load()

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks=text_splitter.split_documents(documents)

    # Create Embeddings using Ollama
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Store chunks in a vector store ( Chroma )
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

    # Initialize the chat Model
    llm = ChatOllama(model="llama2:13b")

    # Create a Prompt Template
    prompt = PromptTemplate(
        template="Use the following context to answer the question: {context}\nQuestion: {question}",
        input_variables=["context", "question"]
    )

    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    # Clean up temporary file
    os.remove("temp.pdf")

    return qa_chain

# Process pdf when file upload
if upload_file is not None and st.session_state.qa_chain is None:
    with st.spinner("Processing the PDF... "):
        st.session_state.qa_chain = process_pdf(upload_file)
    st.success("PDF Uploaded Successfully!")

# Input for asking Questions
question = st.text_input("Ask a question about PDF")

# Button to submit the question
if st.button("Submit", key="submit_with_pdf") and question and st.session_state.qa_chain:
    with st.spinner("Generating answer... "):
        result = st.session_state.qa_chain.invoke({"query": question})
        answer = result["result"]
        st.write("**Answer**", answer)
        st.write("**SourceDocument**", result["source_documents"])

# Clear the session_state to upload a new pdf
if st.button("Clear and Upload a New PDF"):
    st.session_state.qa_chain = None
    st.rerun()

# Instructions
st.sidebar.markdown("""
### Instructions
1. Upload a PDF file using the uploader.
2. Wait for the PDF to be processed.
3. Ask a question in the text box and click "Submit".
4. To upload a new PDF, click "Clear and Upload New PDF" and refresh the page.
""")

if __name__ == "__main__":
    print("Running Streamlit app at http://localhost:8501")
    print("Run command:  streamlit run app.py")