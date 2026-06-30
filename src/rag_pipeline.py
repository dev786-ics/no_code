import time
import os
import warnings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

warnings.filterwarnings("ignore", category=DeprecationWarning)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_PATH = os.path.join(CURRENT_DIR, "vectorstore")

try:
    db = FAISS.load_local(
        VECTOR_STORE_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
except Exception as e:
    raise RuntimeError(f"Failed to load FAISS vector store from {VECTOR_STORE_PATH}. Error: {e}")

llm = Ollama(model="llama3")

def get_answer(query: str) -> dict:
    """
    Retrieves top 3 matching chunks from FAISS, constructs a strict context prompt,
    and returns the local Ollama LLM response with metrics.
    """
    start_time = time.time()
    
    docs = db.similarity_search(query, k=3)
    
    if not docs:
        return {
            "answer": "The answer is not available in the provided document.",
            "sources": [],
            "latency": time.time() - start_time,
            "has_answer": False
        }

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are an AI assistant answering questions ONLY from the given context.

    STRICT RULES:
    - Use ONLY the context below
    - Do NOT use outside knowledge
    - If answer not found, say exactly: "The answer is not available in the provided document."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """
    
    response = llm.invoke(prompt).strip()
    latency = time.time() - start_time
    
    # Evaluate if the model successfully extracted an answer or admitted absence
    has_answer = "not available in the provided document" not in response.lower()

    return {
        "answer": response,
        "sources": [doc.page_content[:200] for doc in docs], # Snippets for reference
        "latency": latency,
        "has_answer": has_answer
    }

if __name__ == "__main__":
    # Isolated test block
    print("--- Running Isolated Pipeline Test ---")
    result = get_answer("What is the AWS Responsibility?")
    print(f"Answer: {result['answer']}")