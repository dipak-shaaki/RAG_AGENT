from dotenv import load_dotenv
load_dotenv() 

from fastapi import FastAPI
from app.routers import chat
from app.routers import ingest

app = FastAPI(
    title="RAG Agent API",
    description="Agent built with LangChain",
    version="1.0"
)

app.include_router(ingest.router)
app.include_router(chat.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG Agent API. Use /ingest to upload documents and /chat to interact."}