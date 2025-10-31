from src.helper import *
# from main import *
import os
import re
# Document Loaders
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
# Text Splitters
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from typing import List
from langchain.schema import Document
# Vector stores
from langchain_community.vectorstores import Pinecone
from pinecone import ServerlessSpec,Pinecone
from langchain_pinecone import PineconeVectorStore
# Retrievers
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
# Prompts
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_core.prompts import PromptTemplate
# Output parser
from langchain_core.output_parsers import StrOutputParser
# Open Source Models
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
# Import runnables
from langchain.schema.runnable import RunnableParallel,RunnablePassthrough,RunnableLambda
# Import prompt
from src.prompt import *
# Loading data from .env files
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, render_template, jsonify, request