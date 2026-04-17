import os
from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_PATH = "chroma_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_retriever = None

def has_vectorstore() -> bool:
    return os.path.exists(CHROMA_PATH)


def reset_retriever_cache() -> None:
    global _retriever
    _retriever = None


def get_retriever():
    """
    Loads Chroma index from disk.
    No allow_dangerous_deserialization needed — Chroma is safe by default.
    """
    if not has_vectorstore():
        raise FileNotFoundError(
            "No Chroma index found. Please call POST /ingest first."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Just point to the folder where Chroma loads automatically
    vectorstore = Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    return vectorstore.as_retriever(search_kwargs={"k": 3})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@tool
def RAG_tool(query: str) -> str:
    """
    Use this tool when the user asks a question about
    documents, reports, policies, files, or knowledge base content.
    Input: the user's question as a plain string.
    """
    print(f"RAG_tool called with query: {query}")
    try:
        retriever = get_retriever()
    except FileNotFoundError:
        print("No index found, returning fallback")
        return "I don't have any information about the document you're referring to. Could you please provide more context or details about the document? I'll do my best to help you understand what it's about."

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        
        """Answer the question using only the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
""")

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke(query)
    print(f"RAG_tool result: {result}")
    return result
