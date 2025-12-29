import os
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from fastapi.concurrency import run_in_threadpool
from ..helper.helper import load_llm, load_embedding_model, get_retriever, create_chain
from ..helper.prompt import template

# Lazy globals
_chain = None

def get_chain():
    global _chain

    if _chain is None:
        print("🚀 Initializing LangChain pipeline...")

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

        _chain = create_chain(retriever, llm, parser, template)

        print("✅ LangChain pipeline initialized")

    return _chain


async def invoke_chain_async(chain, question: str):
    """
    Safely run blocking LangChain call in a threadpool.
    Prevents blocking FastAPI event loop.
    """
    return await run_in_threadpool(chain.invoke, question)
