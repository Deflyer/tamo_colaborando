## Agent in which a human passes continuous feedback to draft a document, until the human is satisifed 

import os
from typing import Annotated, Sequence, TypedDict, Callable, Optional
import difflib
import re

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

def load_pdf_pages(file_path: str, source_name: Optional[str] = None):
    """Load pages from a PDF and annotate each page's metadata with a stable source name.

    If `source_name` is provided, it will be used for `source_file` in metadata. This
    prevents temporary filenames from leaking into chunk metadata when files are
    uploaded and written to temporary paths.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    # Use the provided source_name if given, otherwise the basename of the path
    basename = source_name or os.path.basename(file_path)
    for idx, page in enumerate(pages):
        try:
            meta = page.metadata or {}
        except Exception:
            meta = {}
        # prefer existing page number metadata if present, else use index+1
        page_number = meta.get("page") or meta.get("page_number") or (idx + 1)
        meta["source_file"] = basename
        meta["page_number"] = page_number
        page.metadata = meta
    return pages

def build_vectorstore_from_pages(pages, embeddings, persist_directory: str = "./vdb", collection_name: str = "book"):
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # Split each page individually so that chunk metadata keeps the originating page metadata
    chunks = []
    for page in pages:
        page_chunks = splitter.split_documents([page])
        for ch in page_chunks:
            # ensure chunk metadata contains source_file and page_number
            try:
                ch_meta = ch.metadata or {}
            except Exception:
                ch_meta = {}
            # overlay page metadata (page metadata takes precedence)
            page_meta = getattr(page, "metadata", {}) or {}
            merged = {**ch_meta, **page_meta}
            ch.metadata = merged
            chunks.append(ch)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    return vectorstore


def load_vectorstore_from_persist(persist_directory: str = "./vdb", collection_name: str = "book", embeddings=None):
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Persist directory not found: {persist_directory}")
    if embeddings is None:
        embeddings = build_embeddings()

    vectorstore = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    return vectorstore

def build_retriever(vectorstore, k: int = 7):
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

def build_agent(retriever, llm, history_file: Optional[str] = None):
    @tool
    def retriever_tool(query: str) -> str:
        """Search and return relevant chunks from the loaded PDF.

        The returned text includes simple citations (source file and page number)
        alongside the chunk content to help the LLM cite the origin.
        """
        source_name = None
        search_query = query
        try:
            m = re.search(r"source\s*:\s*([^\n,;]+)", query, flags=re.IGNORECASE)
            if not m:
                m = re.search(r"from\s+[\"']?([^\n\"'.,;]+)[\"']?", query, flags=re.IGNORECASE)
            if m:
                source_name = m.group(1).strip()
                search_query = (query[:m.start()] + query[m.end():]).strip()
        except Exception:
            source_name = None

        docs = retriever.invoke(search_query)
        if not docs:
            return "No relevant info was found in the document"
        results = []
        available_sources = []
        for d in docs:
            meta = getattr(d, "metadata", {}) or {}
            src = meta.get("source_file", meta.get("source"))
            if src and src not in available_sources:
                available_sources.append(src)

        filtered_docs = docs
        matched_source = None
        if source_name:
            matches = difflib.get_close_matches(source_name, available_sources, n=1, cutoff=0.6)
            if matches:
                matched_source = matches[0]
                filtered_docs = [d for d in docs if (getattr(d, "metadata", {}) or {}).get("source_file") == matched_source]
            else:
                lowered = {s.lower(): s for s in available_sources}
                if source_name.lower() in lowered:
                    matched_source = lowered[source_name.lower()]
                    filtered_docs = [d for d in docs if (getattr(d, "metadata", {}) or {}).get("source_file") == matched_source]
                else:
                    return f"No document found matching '{source_name}'. Available: {', '.join(available_sources) if available_sources else 'none'}"

        for i, doc in enumerate(filtered_docs):
            meta = getattr(doc, "metadata", {}) or {}
            source = meta.get("source_file", meta.get("source", "unknown"))
            page_no = meta.get("page_number", meta.get("page", "?"))
            snippet = doc.page_content.strip()
            citation = f"(source: {source}, page: {page_no})"
            results.append(f"Document {i+1} {citation}:\n{snippet}")
        if matched_source:
            note = f"[Filtered to source: {matched_source}]\n\n"
        else:
            note = ""
        return "\n\n".join(results)
    # Criar tool para trabalhar com o historico da conversa
    # Criar tool para gerar questões novas
    # Criar tool para gerar resumos
    @tool
    def conversation_history_tool(query: str) -> str:
        """Return the most recent N messages from a conversation history text file.

        The query can be an integer (as string) indicating how many recent messages to return.
        Defaults to 20 and will never return more than 20 messages.
        """
        path = history_file or os.environ.get("RAG_HISTORY_FILE") or os.path.join("./vdb", "conversation_history.txt")
        try:
            n = int(query.strip()) if query and query.strip().isdigit() else 20
        except Exception:
            n = 20
        n = max(0, min(20, n))
        if not os.path.exists(path):
            return "No conversation history found."
        try:
            with open(path, "r", encoding="utf-8") as fh:
                lines = fh.read().splitlines()
        except Exception as e:
            return f"Error reading history file: {e}"

        last_lines = lines[-n:]
        return "\n".join(last_lines) if last_lines else "No conversation history available."

    tools = [retriever_tool, conversation_history_tool]
    llm_with_tools = llm.bind_tools(tools)

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def should_continue(state: AgentState):
        # Inspect the last message's tool calls and return the tool name
        last = state["messages"][-1]
        if not hasattr(last, "tool_calls") or len(last.tool_calls) == 0:
            return False
        # return the first tool name called (graph can decide routing based on this)
        tool_name = last.tool_calls[0].get("name") if isinstance(last.tool_calls[0], dict) else getattr(last.tool_calls[0], "name", None)
        return tool_name or False

    system_prompt = (
        "You are an assistant that answers questions about the PDF loaded into the knowledge base. "
        "When the information relies on retrieved document content, include concise citations to the specific parts of the document (e.g. filename and page number). "
        "Only add citations when they materially support the answer or when directly relevant — avoid adding citations for trivial or speculative remarks. "
        "If you quote or paraphrase a retrieved snippet, append a short citation in parentheses after the relevant sentence (format: source: <filename>, page: <n>)."
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
    graph.add_node("history", take_action)

    # Route based on which tool was called.
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            "retriever_tool": "retriever",
            "conversation_history_tool": "history",
            False: END,
        },
    )
    graph.add_edge("retriever", "llm")
    graph.add_edge("history", "llm")
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
