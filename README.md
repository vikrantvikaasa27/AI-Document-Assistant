# LangChain AI Doc Assistant

## Overview
LangChain AI Doc Assistant is an intelligent document processing application built using Streamlit and LangChain. It allows users to upload research documents (PDFs), process them, and ask questions based on their content. The system uses the NVIDIA AI endpoint (Llama 3.3-70B) for natural language processing and provides concise, factual responses.

## Features
- Upload and analyze PDF documents.
- Extract relevant document content using LangChain's PDFPlumberLoader.
- Chunk text for efficient processing.
- Use In-Memory Vector Store for document embeddings and similarity search.
- Generate intelligent responses using NVIDIA's Llama-3.3-70B-instruct model.
- Interactive chat UI built with Streamlit.

## Installation
### Prerequisites
Ensure you have an NVIDIA API key to access the Llama-3.3-70B-instruct model.
If you don't have one, create one from here https://build.nvidia.com/explore/discover

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/langchain-ai-doc-assistant.git
   cd langchain-ai-doc-assistant
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add your NVIDIA API key:
     ```
     NVIDIA_API_KEY=your_nvidia_api_key
     ```
5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the app in your browser (default: `http://localhost:8501`).
2. Upload a PDF document.
3. Wait for the document to be processed.
4. Ask questions related to the document's content.
5. View AI-generated responses based on the document context.

## Technologies Used
- **LangChain**: Framework for building LLM-powered applications.
- **Streamlit**: Web-based UI for document interaction.
- **NVIDIA AI Endpoints**: Llama-3.3-70B-instruct model for question-answering.
- **PDFPlumber**: Extracting text from PDFs.
- **RecursiveCharacterTextSplitter**: Chunking document text for processing.
- **InMemoryVectorStore**: Storing and retrieving document embeddings.

## Future Enhancements
- Implement database support for storing document embeddings.
- Enable multi-document uploads and cross-document querying.
- Support additional LLM models and APIs.

## Contributing
Contributions are welcome! Feel free to fork the repository, create a new branch, and submit a pull request.


