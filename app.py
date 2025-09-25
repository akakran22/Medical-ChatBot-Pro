import os
import time
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session
from dotenv import load_dotenv

from utils.read_preprocess import DocumentProcessor
from utils.chunk_data import TextChunker
from utils.qdrant_db import VectorDatabase
from utils.retrieval_qa import LLMAgent
from utils.tavily import WebScraper
from utils.critic_agent import CriticAgent

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key')

# Initialize components
vector_db = VectorDatabase()
llm_agent = LLMAgent()
web_scraper = WebScraper()
critic_agent = CriticAgent()
doc_processor = DocumentProcessor()
text_chunker = TextChunker()


def initialize_database():
    try:
        print("Checking database status...")
        
        # Check if collection has documents
        count = vector_db.get_collection_count()
        if count > 0:
            print(f" Using existing collection with {count} documents")
            return True
        
        # Collection is empty or doesn't exist, need to load documents
        print("Loading and processing PDF documents...")
        
        # Load documents
        documents = doc_processor.get_all_documents()
        if not documents:
            print(" No PDF documents found in data folder")
            return False
        
        print(f"Found {len(documents)} PDF documents. Processing...")
        
        # Chunk documents
        chunked_docs = text_chunker.chunk_documents(documents)
        if not chunked_docs:
            print(" No chunks created from documents")
            return False
        
        print(f"Created {len(chunked_docs)} chunks. Storing in vector database...")
        
        # Store in database
        success = vector_db.store_documents(chunked_docs)
        if success:
            final_count = vector_db.get_collection_count()
            print(f" Database initialized successfully with {final_count} documents")
            return True
        else:
            print(" Failed to store documents in vector database")
            return False
            
    except Exception as e:
        print(f" Database initialization error: {e}")
        return False


def process_medical_query(query):
    start_time = time.time()
    try:
        # Step 1: Vector search
        vector_results = vector_db.search_similar(query, limit=5)

        # Step 2: Web search
        web_results = web_scraper.search_web(query, max_results=3)

        # Step 3: Generate response
        llm_response = llm_agent.generate_response(query, vector_results, web_results)

        # Step 4: Critic evaluation
        critic_eval = critic_agent.evaluate_response(query, llm_response, vector_results, web_results)
        critic_score = critic_eval.get("score", 0)

        final_response = llm_response
        if critic_eval.get("needs_more_info", False) and critic_score < 6:
            additional_web = web_scraper.search_web(
                f"{query} detailed medical information treatment", max_results=2
            )
            web_results.extend(additional_web)
            final_response = llm_agent.generate_response(query, vector_results, web_results)

        return {
            "query": query,
            "vector_results": vector_results,
            "web_results": web_results,
            "llm_response": llm_response,
            "final_response": final_response,
            "critic_score": critic_score,
            "processing_time": time.time() - start_time,
        }
    except Exception as e:
        print(f"Error processing query: {e}")
        return {
            "query": query,
            "vector_results": [],
            "web_results": [],
            "llm_response": "I encountered an error while processing your query.",
            "final_response": "I encountered an error while processing your query.",
            "critic_score": 0,
            "processing_time": time.time() - start_time,
        }


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    query = request.form.get("query", "").strip()
    if not query:
        flash("Please enter a medical question.", "error")
        return redirect(url_for("index"))

    result = process_medical_query(query)

    if "chat_history" not in session:
        session["chat_history"] = []

    session["chat_history"].append(
        {
            "query": query,
            "response": result["final_response"],
            "score": result["critic_score"],
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
    )
    session["chat_history"] = session["chat_history"][-20:]
    session.modified = True

    return render_template("chat.html", result=result)


@app.route("/clear_history")
def clear_history():
    session.pop("chat_history", None)
    flash("Chat history cleared.", "success")
    return redirect(url_for("index"))


@app.route("/status")
def status():
    try:
        count = vector_db.get_collection_count()
        return {"status": "healthy", "documents": count, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    print(" Starting Medical AI Chatbot...")

    # Check required environment variables
    required_keys = ["QDRANT_URL", "QDRANT_API_KEY", "GROQ_API_KEY", "TAVILY_API_KEY", "COHERE_API_KEY"]
    missing = [k for k in required_keys if not os.getenv(k)]
    if missing:
        print(f" Missing environment variables: {', '.join(missing)}")
        exit(1)

    # Initialize database
    if initialize_database():
        print(" Database ready")
    else:
        print(" Database initialization incomplete, but continuing...")

    # Start Flask app
    app.run(
        debug=os.getenv("FLASK_DEBUG", "False").lower() == "true",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
    )