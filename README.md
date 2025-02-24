## PDF Question-Answer System
A simple Streamlit web app to upload PDF files, process them, and answer questions based on their content using LangChain and Ollama.

## Features
* Upload a PDF file.
* Ask questions about the PDF content.
* Get answers powered by a retrieval-based QA system.
* Clear and upload a new PDF anytime.
## Requirements
* Python 3.8+
* Libraries:
  * streamlit
  * langchain
  * langchain-community
  * langchain-ollama
  * pypdf
  * chroma
## Installation
### Clone this repository:
**bash**:

git clone <repository-url>

cd <repository-folder>

### Install dependencies:
**bash**:

pip install -r requirements.txt

* Ensure Ollama is installed and running locally with the **nomic-embed-text** and **llama2:13b** models.
* Download Ollama in Local.
* Terminal : ollama pull  **nomic-embed-text** , **llama2:13b**
## Usage
### Run the app:
**bash**:

streamlit run app.py

Open your browser at http://localhost:8501.

**Upload a PDF**, wait for processing, and **ask questions** in the text box.
## How It Works
* Uploads a PDF and saves it temporarily.
* Splits the PDF into chunks using **RecursiveCharacterTextSplitter.**
* Embeds chunks with **OllamaEmbeddings** and stores them in a Chroma vector store.
* Uses **RetrievalQA** with ChatOllama to answer questions based on the PDF content.
## Notes
* Temporary files are cleaned up after processing.
* The app stores the QA chain in Streamlit’s session state for efficiency.
* Requires a local Ollama instance.
## License
This project is licensed under the MIT License. See the LICENSE file for more details.