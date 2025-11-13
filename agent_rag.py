## Agent in which a human passes continuous feedback to draft a document, until the human is satisifed 

import os
from typing import Annotated, Sequence, TypedDict, Callable, Optional

from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

from operator import add as add_messages
from dotenv import load_dotenv

def build_llm(model: str = "nvidia/nemotron-nano-12b-v2-vl:free", temperature: float = 0):
    llm = ChatOpenAI(
        model=model,
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature
    )
    return llm

def build_embeddings(embedding_model_name: str = "all-MiniLM-L6-v2"):
    class SentenceTransformerEmbeddings:
        def __init__(self, model_name):
            self.model = SentenceTransformer(model_name, device="cpu")

        def embed_documents(self, texts):
            return self.model.encode(texts, convert_to_tensor=False)

        def embed_query(self, text):
            return self.model.encode([text], convert_to_tensor=False)[0]

    return SentenceTransformerEmbeddings(embedding_model_name)

def load_pdf_pages(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages

def build_vectorstore_from_pages(pages, embeddings, persist_directory: str = "./vdb", collection_name: str = "book"):
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    return vectorstore

def build_retriever(vectorstore, k: int = 7):
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

def build_agent(retriever, llm):
    @tool
    def retriever_tool(query: str) -> str:
        """Search and return relevant chunks from the loaded PDF"""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant info was found in the document"
        results = []
        for i, doc in enumerate(docs):
            results.append(f"Document {i+1}:\n{doc.page_content}")
        return "\n\n".join(results)
    # Criar tool para trabalhar com o historico da conversa
    # Criar tool para gerar questões novas
    # Criar tool para gerar resumos
    tools = [retriever_tool]
    llm_with_tools = llm.bind_tools(tools)

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        return hasattr(last, "tool_calls") and len(last.tool_calls) > 0

    system_prompt = (
        "You are an assistant that answers questions about the PDF loaded into the knowledge base. "
        "Verify information with citations to the specific parts of the document."
    )

    tools_dict = {t.name: t for t in tools}

    def call_llm(state: AgentState) -> AgentState:
        msgs = list(state["messages"])
        msgs = [SystemMessage(content=system_prompt)] + msgs
        message = llm_with_tools.invoke(msgs)
        return {"messages": [message]}

    def take_action(state: AgentState) -> AgentState:
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            tool_name = t["name"]
            args_query = t["args"].get("query", "")
            if tool_name not in tools_dict:
                result = "Incorrect tool name; select an available tool and try again."
            else:
                result = tools_dict[tool_name].invoke(args_query)
            results.append(ToolMessage(tool_call_id=t["id"], name=tool_name, content=str(result)))
        return {"messages": results}

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever", take_action)
    graph.add_conditional_edges("llm", should_continue, {True: "retriever", False: END})
    graph.add_edge("retriever", "llm")
    graph.set_entry_point("llm")
    return graph.compile()


def run_rag_agent_cli(file_path: str = "file.pdf", persist_directory: str = "./vdb"):
    load_dotenv()
    llm = build_llm()
    embeddings = build_embeddings()
    pages = load_pdf_pages(file_path)
    vectorstore = build_vectorstore_from_pages(pages, embeddings, persist_directory=persist_directory)
    retriever = build_retriever(vectorstore)
    agent = build_agent(retriever, llm)

    print("======= RAG AGENT ======")
    while True:
        user_input = input("\nQuestion: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        messages = [HumanMessage(content=user_input)]
        result = agent.invoke({"messages": messages})
        print("\n==== ANSWER =====")
        print(result["messages"][-1].content)

if __name__ == "__main__":
    run_rag_agent_cli()
