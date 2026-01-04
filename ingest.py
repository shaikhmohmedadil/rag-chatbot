from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"  # Disable ChromaDB telemetry

# Load environment variables
load_dotenv()

def ingest_documents():
    """Load documents and create vector store"""
    
    print("Loading documents...")
    # Load the text file
    loader = TextLoader("data/sample.txt")
    documents = loader.load()
    
    print("Splitting text...")
    # Split into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    print("Creating embeddings...")
    # Create embeddings
    embeddings = OpenAIEmbeddings()
    
    print("Creating vector store...")
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print("✅ Documents ingested successfully!")
    return vectorstore

if __name__ == "__main__":
    ingest_documents()