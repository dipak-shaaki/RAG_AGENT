import asyncio
from fastapi import APIRouter
from pydantic import BaseModel
from app.agent.agent_executor import create_agent_executor
from app.agent.tools.RAG_tool import RAG_tool, has_vectorstore

router = APIRouter()

DOCUMENT_KEYWORDS = {
    "document",
    "doc",
    "pdf",
    "file",
    "report",
    "policy",
    "manual",
    "resume",
    "cv",
    "paper",
    "contract",
    "invoice",
    "uploaded",
    "upload",
}


def should_use_rag(message: str) -> bool:
    normalized = message.lower()

    if any(keyword in normalized for keyword in DOCUMENT_KEYWORDS):
        return True

    # Treat summary/explanation requests as document questions when a file exists.
    if len(normalized.split()) >= 4:
        return any(
            phrase in normalized
            for phrase in (
                "what is this about",
                "summarize this",
                "summarise this",
                "explain this",
                "from the document",
                "in the document",
            )
        )

    return False

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if has_vectorstore() and should_use_rag(req.message):
        reply = await asyncio.to_thread(RAG_tool.invoke, req.message)
        return ChatResponse(reply=reply)

    agent, config = create_agent_executor(req.session_id)

    result = await asyncio.to_thread(
        agent.invoke,
        {"messages": [{"role": "user", "content": req.message}]},
        config
    )

    reply = result["messages"][-1].content
    return ChatResponse(reply=reply)
