import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)

import tempfile
import os

st.set_page_config(page_title="Mini RAG System", page_icon="📚")

st.title("📚 Mini RAG System")
st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    with st.spinner("Loading PDF..."):

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        docs = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(
            docs,
            embeddings
        )

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

        model_name = "google/flan-t5-base"

        tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        )

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=256
        )

        llm = HuggingFacePipeline(
            pipeline=pipe
        )

        template = """
Use the following context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer:
"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    st.success("PDF processed successfully!")

    query = st.text_input(
        "Ask a question about the document"
    )

    if st.button("Ask") and query:

        with st.spinner("Generating answer..."):

            result = qa_chain(
                {"query": query}
            )

            st.subheader("Answer")
            st.write(result["result"])

            with st.expander("Source Documents"):
                for i, doc in enumerate(
                    result["source_documents"]
                ):
                    st.write(
                        f"Source {i+1}"
                    )
                    st.write(
                        doc.page_content[:500]
                    )

    os.remove(pdf_path)
