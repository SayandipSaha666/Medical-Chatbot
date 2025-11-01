from src.helper import *
from langchain_pinecone import PineconeVectorStore
# Output parser
from langchain_core.output_parsers import StrOutputParser
# Import prompt
from src.prompt import *
from flask import Flask, render_template, jsonify, request


app = Flask(__name__)

model = "openai/gpt-oss-120b"
llm = load_llm(model)

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embedding = load_embedding_model(embedding_model)

index_name="medical-chatbot" 
vector_store = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

retriever = get_retriever(vector_store,llm)
parser = StrOutputParser()

chain = create_chain(retriever,llm,parser,template)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = chain.invoke(msg)
    formatted_response = markdown_to_text(response)
    print("Response : ", formatted_response)
    return str(formatted_response)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= False)