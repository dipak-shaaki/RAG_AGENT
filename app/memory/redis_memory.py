import os
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory

def get_memory(session_id: str) -> ConversationBufferMemory:
    """
    session_id = unique identifier per user or conversation.
    
    What happens:
    - RedisChatMessageHistory stores messages in Redis under the key session_id
    - Each new message is appended to that list
    - ConversationBufferMemory wraps it so the agent can read prior turns
    - ttl=3600 = messages auto-delete after 1 hour of inactivity
    """
    history = RedisChatMessageHistory(
        session_id=session_id,
        url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        ttl=3600
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",   # agent reads history from this key
        chat_memory=history,
        return_messages=True         # keep as message objects (HumanMessage, AIMessage)
    )

    return memory