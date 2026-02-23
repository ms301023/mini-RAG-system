Mini RAG System – Document Question Answering
Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline using LangChain, FAISS, and HuggingFace models to enable context-grounded question answering over PDF documents.

Architecture

Load PDF document

Chunk text using RecursiveCharacterTextSplitter

Generate embeddings using sentence-transformers

Store vectors in FAISS index

Retrieve top-k relevant chunks

Generate grounded response using LLM

Tech Stack

Python

LangChain

FAISS

Sentence-Transformers

HuggingFace (FLAN-T5)

Features

Semantic search using vector similarity

Top-k retrieval tuning

Custom prompt template for hallucination reduction

Source document display
