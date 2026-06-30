import streamlit as st
import requests

st.title("AWS RAG Chatbot & Analytics Dashboard")

st.header("Ask the Document")
query = st.text_input("Enter your question:")

if st.button("Submit Question"):
    if query:
        with st.spinner("Processing your request..."):
            try:
                # Send question to your FastAPI backend
                res = requests.post(
                    "http://127.0.0.1:8000/ask", 
                    json={"query": query}
                )
                
                if res.status_code == 200:
                    data = res.json()
                    
                    # Display the AI's generated response
                    st.write("Answer")
                    st.write(data["answer"])
                    
                    # Display the document context snippets
                    st.write("Reference Sources")
                    for chunk in data.get("sources", []):
                        st.write(f"- {chunk}")
                else:
                    # If FastAPI breaks, display the exact text error inside a block
                    st.error(f"Backend Error (Status {res.status_code})")
                    st.code(res.text)
                    
            except Exception as e:
                st.error(f"Cannot connect to FastAPI server. Is it running? Error: {e}")

st.write("---")
st.header("Usage Analytics Dashboard")

if st.button("Refresh System Metrics"):
    try:
        # Request aggregate logs from FastAPI analytics route
        res = requests.get("http://127.0.0.1:8000/analytics")
        
        if res.status_code == 200:
            metrics = res.json()
            
            # Simple layout columns for the core metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Queries Processed", metrics["total_queries"])
            col2.metric("Unanswered Queries Count", metrics["unanswered_queries"])
            col3.metric("Average Latency (Seconds)", f"{metrics['average_latency_seconds']}s")
            
            # Simple presentation table for top questions
            st.write("Top 5 Most Frequent Queries")
            if metrics["frequent_queries"]:
                st.write(metrics["frequent_queries"])
            else:
                st.write("No system logs recorded yet.")
        else:
            st.error(f"Analytics Endpoint Error (Status {res.status_code})")
            st.code(res.text)
            
    except Exception as e:
        st.error(f"Could not connect to Analytics Backend: {e}")

