# ------------ To automate creation of Pinecone vector DataBase -----------------


from src.helper import *
import os
# Output parser
from langchain_core.output_parsers import StrOutputParser
from src.prompt import *
# Loading data from .env files
from dotenv import load_dotenv
load_dotenv()


model = "openai/gpt-oss-120b"
llm = load_llm(model)
parser = StrOutputParser()

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embedding = load_embedding_model(embedding_model)

docs = load_pdf(data='data/')
filtered_docs = filter_documents(docs)

chunks = chunk_text_semantically(filtered_docs,embedding)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

vector_store = create_database(index_name="medical-chatbot",chunks=chunks,embedding_model=embedding,api_key=PINECONE_API_KEY)


