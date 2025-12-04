from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import os
from src.helper import *
from src.prompt import *
from langchain_core.output_parsers import StrOutputParser
import markdown
from pymongo import MongoClient
app = Flask(__name__)
MONGO_API_KEY = os.getenv("MONGO_API_KEY")
client = MongoClient(f"mongodb+srv://sahasbhs2022_db_user:{MONGO_API_KEY}@cluster0.wyh2zre.mongodb.net/")
db = client["MedGPT"] 
# Enable CORS for all routes - allows frontend to communicate with backend
CORS(app, 
     origins="*",
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type"])

# Lazy globals (initialized on first request only)

llm = None
embedding = None
vector_store = None
retriever = None
parser = None
chain = None

def initialize_chain():
    """Initialize heavy LangChain components lazily and cache them."""
    global llm, embedding, vector_store, retriever, parser, chain
    

    if chain is None:  # Only build once
        print(" Initializing LangChain pipeline...")
        print("Ping:", client.admin.command("ping"))
        model = "openai/gpt-oss-120b"
        llm = load_llm(model)

        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        embedding = load_embedding_model(embedding_model)

        index_name = "medical-chatbot"
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embedding
        )

        retriever = get_retriever(vector_store, llm)
        parser = StrOutputParser()
        chain = create_chain(retriever, llm, parser, template)

        print(" LangChain pipeline initialized.")
    return chain


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint to verify backend is running"""
    return jsonify({
        "status": "ok",
        "message": "MedGPT backend is running",
        "chain_initialized": chain is not None
    }), 200


@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        # Get message from request
        if request.method == "POST":
            msg = request.form.get("msg", "").strip()
        else:
            msg = request.args.get("msg", "").strip()
        
        print(f"[REQUEST] Method: {request.method}, Message: {msg}")
        
        # Validate message
        if not msg:
            error_msg = "Message cannot be empty"
            print(f"[ERROR] {error_msg}")
            return jsonify({"error": error_msg}), 400
        
        print("[PROCESSING] Initializing chain...")
        
        # Initialize and get chain
        chain = initialize_chain()
        
        print("[PROCESSING] Invoking chain with message...")
        
        # Generate response
        response = chain.invoke(msg)
        
        print(f"[RESPONSE] Generated response: {response[:100]}...")
        
        # Convert markdown to HTML
        formatted_response = markdown.markdown(response, extensions=['fenced_code', 'tables'])
        
        print(f"[SUCCESS] Returning formatted response")
        
        # Return as plain text (frontend will handle HTML)
        return formatted_response, 200, {'Content-Type': 'text/html; charset=utf-8'}
    
    except Exception as e:
        error_msg = f"Error in chat endpoint: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)