import os
import openai
import faiss
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

app = FastAPI()

openai.api_key = os.getenv('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4")

FAISS_INDEX_DIR = "faiss_index"

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_path = f"{uploaded_pdf_file_path}"
    
    with open(file_path, "wb") as f:
        f.write(await file.read())

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    vector_store = FAISS.from_documents(texts, embeddings)

    vector_store.save_local(FAISS_INDEX_DIR)

    return JSONResponse(content={"message": "PDF processed and FAISS index created successfully"}, status_code=200)

from pydantic import BaseModel

class QueryInput(BaseModel):
    query: str

@app.post("/retrieve")
async def retrieve_answer(query_input: QueryInput):
    query = query_input.query
    if not query:
        raise HTTPException(status_code=400, detail="No query provided")

    vector_store = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())

    answer = qa_chain.run(query)

    return JSONResponse(content={"answer": answer})
