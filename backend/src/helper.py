import re
# Document Loaders
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
# Text Splitters
from langchain_experimental.text_splitter import SemanticChunker
from typing import List
from langchain.schema import Document
# Vector stores
from langchain_community.vectorstores import Pinecone
from pinecone import ServerlessSpec,Pinecone
from langchain_pinecone import PineconeVectorStore
# Retrievers
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
# Open Source Models
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
# Import runnables
from langchain.schema.runnable import RunnableParallel,RunnablePassthrough,RunnableLambda
# Loading data from .env files
from dotenv import load_dotenv
load_dotenv()

# Load LLM
def load_llm(model):
    llm = ChatGroq(
        model=model
    )
    return llm

# Load Embedding Model
def load_embedding_model(embedding_model):
    embedding = HuggingFaceEmbeddings(
        model_name=embedding_model,
        encode_kwargs={'normalize_embeddings': True}
    )
    return embedding

# Load PDF Files
def load_pdf(folder):
    loader = DirectoryLoader(
        folder,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    docs = loader.load()
    return docs


def filter_documents(docs):
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={'source':src}
            )
        )
    return minimal_docs


def chunk_text_semantically(docs,embedding_model):
    # use semantic chunker
    splitter = SemanticChunker(
        embedding_model,breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=1
    )
    chunks = splitter.split_documents(docs)
    return chunks


def create_database(index_name,chunks,embedding_model,api_key):
    pc = Pinecone(api_key=api_key)
    index_name=index_name
    if not pc.has_index(index_name):
        pc.create_index(
            name = index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws",region="us-east-1")
        )
    index = pc.Index(index_name)
    docsearch = PineconeVectorStore.from_documents(
        documents = chunks,
        embedding = embedding_model,
        index_name=index_name
    )   
    return docsearch


def get_retriever(vector_store,llm):
    similarity_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k":3}
    )
    compressor = LLMChainExtractor.from_llm(llm)
    retriever = ContextualCompressionRetriever(
        base_retriever=similarity_retriever,
        base_compressor=compressor
    )
    return retriever

def create_chain(retriever,llm,parser,template):

    parallel_chain = RunnableParallel({
        "question": RunnablePassthrough(),
        "context": retriever | RunnableLambda(lambda context: "\n\n".join(doc.page_content for doc in context))
    })

    chain = parallel_chain | template | llm | parser 

    return chain

def markdown_to_text(md_text: str) -> str:
    # Remove bold and italics
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', md_text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # Replace bullet points
    text = text.replace("- ", "• ")
    # Replace escaped newlines
    text = text.replace("\\n", "\n")
    return text.strip()