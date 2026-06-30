import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_pipeline import get_answer
from db import init_db, insert_log, DB_NAME

app = FastAPI(title="AWS RAG Analytics Backend")

init_db()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    try:
        result = get_answer(request.query)
        
        insert_log(
            query=request.query, 
            answer=result["answer"], 
            latency=result["latency"], 
            has_answer=result["has_answer"]
        )

        return {
            "query": request.query,
            "answer": result["answer"],
            "sources": result["sources"]
        }
    except Exception as e:
        # Throw standard HTTP 500 instead of crashing silently
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

@app.get("/analytics")
def get_analytics():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute("SELECT AVG(latency) FROM logs")
        avg_latency = cursor.fetchone()[0] or 0.0
        
        cursor.execute("SELECT COUNT(*), SUM(CASE WHEN has_answer = 0 THEN 1 ELSE 0 END) FROM logs")
        total, unanswered = cursor.fetchone()
        total = total or 0
        unanswered = unanswered or 0

        cursor.execute("SELECT query, COUNT(query) as c FROM logs GROUP BY query ORDER BY c DESC LIMIT 5")
        freq_queries = [{"query": row[0], "count": row[1]} for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "total_queries": total,
            "unanswered_queries": unanswered,
            "average_latency_seconds": round(avg_latency, 3),
            "frequent_queries": freq_queries
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Analytics Error: {str(e)}")