# Import required libraries

from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

path = "C:\\Users\\anjan\\Desktop\\work_space\\rag_work\\Rag_chatbot\\no_code\\aws_work\\main\\data\\AWS Customer Agreement.pdf"
def ingest_pdf(path):

    # Load PDF

    loader = PyPDFLoader(path)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    # Initialize BGE embeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en"
    )

    #  Semantic Chunking

    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile"
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} semantic chunks")

    # FAISS vector 

    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save vector store locally
    
    vector_store.save_local("vectorstore")

if __name__ == "__main__":
    ingest_pdf(path)