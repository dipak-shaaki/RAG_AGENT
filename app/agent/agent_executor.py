import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from app.agent.tools.RAG_tool import RAG_tool
from app.agent.tools.booking_tool import booking_tool

TOOLS = [RAG_tool, booking_tool]
checkpointer = MemorySaver()

def create_agent_executor(session_id: str):
    api_key = os.getenv("GROQ_API_KEY")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.3
    )

    agent = create_agent(
        model=llm,
        tools=TOOLS,
        checkpointer=checkpointer,
        system_prompt=(
            "You are a helpful assistant with access to tools for specific tasks. "
            "Always analyze the user's intent and use the appropriate tool: "
            "- Use RAG_tool ONLY when the user asks questions about uploaded documents, reports, policies, files, or knowledge base content that has been ingested. "
            "- Use booking_tool ONLY when the user asks about making bookings, appointments, reservations, or scheduling. "
            "- For any other questions, casual conversation, or topics not related to documents or booking, respond directly without using any tools. "
            "Do not use tools for general knowledge questions or when no relevant content has been uploaded."
        )
    )

    config = {"configurable": {"thread_id": session_id}}
    return agent, config