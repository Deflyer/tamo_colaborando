import os
import time
import uuid
import tempfile
import streamlit as st
from dotenv import load_dotenv

from agent_rag import (
    build_llm,
    build_embeddings,
    build_vectorstore_from_pages,
    build_retriever,
    build_agent,
    load_pdf_pages,
    load_vectorstore_from_persist,
)

USERS = ["Artur", "Pedro", "JP", "Rebecca", "John Doe"]

def get_shared_vectorstore_dir() -> str:
    base = os.environ.get("RAG_VDB_DIR", "./vdb")
    if not os.path.exists(base):
        os.makedirs(base)
    return base


def ensure_session_state():
    """Initialize all session variables"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {role: "user"|"assistant"|"system", content: str}
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "selected_user" not in st.session_state:
        st.session_state.selected_user = USERS[0]


def get_history_file_path() -> str:
    return os.path.join(get_shared_vectorstore_dir(), "conversation_history.txt")


def get_index_registry_path() -> str:
    return os.path.join(get_shared_vectorstore_dir(), "indexed_files.txt")


def read_indexed_files() -> list:
    path = get_index_registry_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh.read().splitlines() if l.strip()]
        return lines
    except Exception:
        return []


def add_indexed_file(filename: str):
    path = get_index_registry_path()
    files = set(read_indexed_files())
    if filename in files:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{filename}\n")
    except Exception:
        pass


def append_history_to_file(message: dict):
    path = get_history_file_path()
    try:
        role = message.get("role", "")
        user = message.get("user", "")
        content = message.get("content", "").replace("\n", " ")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{role}::{user}::{content}\n")
    except Exception as e:
        st.error(f"Failed to write history: {e}")

def build_or_update_index_from_uploads(uploaded_files):
    """Build or update the vectorstore from a list of uploaded files.

    `uploaded_files` is expected to be a list of Streamlit UploadedFile objects.
    The function writes each upload to a temp file, loads pages using the original
    filename as `source_name` (so metadata keeps the real filename), indexes all
    pages together, and registers the filenames in `indexed_files.txt`.
    """
    temp_paths = []
    all_pages = []

    # Normalize single-file case
    if uploaded_files is None:
        return None
    if not isinstance(uploaded_files, (list, tuple)):
        uploaded_files = [uploaded_files]

    try:
        for f in uploaded_files:
            # `f` can be a Streamlit UploadedFile-like object with .read() and .name
            filename = getattr(f, "name", None)
            content = f.read() if hasattr(f, "read") else f
            suffix = os.path.splitext(filename or "")[1] or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            temp_paths.append(tmp_path)
            # load pages with correct source name and extend
            all_pages.extend(load_pdf_pages(tmp_path, source_name=filename))
            # register the indexed filename so the UI can show it
            if filename:
                add_indexed_file(filename)

        embeddings = build_embeddings()
        vectorstore = build_vectorstore_from_pages(
            all_pages,
            embeddings,
            persist_directory=get_shared_vectorstore_dir(),
            collection_name="book",
        )
        retriever = build_retriever(vectorstore)
        return retriever
    finally:
        # cleanup temp files
        for p in temp_paths:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass


def main():
    load_dotenv()
    st.set_page_config(page_title="RAG Chat (aula)", page_icon="📄")
    ensure_session_state()

    # Auto-connect to an existing persisted vectorstore if present
    shared_dir = get_shared_vectorstore_dir()
    try:
        if st.session_state.retriever is None and os.path.exists(shared_dir) and os.listdir(shared_dir):
            with st.spinner("Connecting to existing vectorstore..."):
                embeddings = build_embeddings()
                vectorstore = load_vectorstore_from_persist(persist_directory=shared_dir, collection_name="book", embeddings=embeddings)
                st.session_state.retriever = build_retriever(vectorstore)
            st.success("Connected to existing vectorstore.")
    except Exception as e:
        st.warning(f"Could not connect to existing vectorstore: {e}")

    st.title("📄 RAG Chat (aula)")
    st.caption("Upload a PDF, then chat. Multiple users share the same vector store.")

    with st.sidebar:

        st.header("User")
        st.session_state.selected_user = st.selectbox("Active user", USERS, index=USERS.index(st.session_state.selected_user))
        st.caption("Messages sent will be attributed to this user.")

        st.header("Data")
        # upload PDF(s) - accept multiple files
        uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
        # if success, allow button to build/update vectorDB index
        if uploaded_files:
            if st.button("Build/Update Index", type="primary"):
                with st.spinner("Creating index... "):
                    st.session_state.retriever = build_or_update_index_from_uploads(uploaded_files)
                st.success("Index created from uploaded files.")
        
        st.divider()
        # Show indexed files found in the shared vectorstore directory
        st.header("Indexed Files")
        indexed = read_indexed_files()
        if indexed:
            for fname in indexed:
                st.markdown(f"- `{fname}`")
        else:
            st.markdown("_No indexed files found._")

        st.header("Agent")
        # slider k
        k = st.slider("Retrieve k chunks", min_value=2, max_value=10, value=6, step=2)

        # button for agent creation
        if st.button("(re)Create Agent"):
            if st.session_state.retriever is None:
                st.warning("Please build the index first")
            else:
                llm = build_llm(temperature=0)
                retriever = st.session_state.retriever
                retriever.search_kwargs["k"] = k
                st.session_state.agent = build_agent(retriever, llm)
                st.success("Agent ready.")

    # Chat area
    st.subheader("Espaço para discussão entre os usuários, caso deseje chamar o agente, marque o com @colaborai na mensagem")
    for msg in st.session_state.messages:
        author = msg.get("user", "User") if msg["role"] == "user" else "assistant"
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(f"**{author}**: {msg['content']}")

    if prompt := st.chat_input(f"{st.session_state.selected_user} diz: "):
        user_msg = {"user": st.session_state.selected_user, "role": "user", "content": prompt}
        st.session_state.messages.append(user_msg)
        append_history_to_file(user_msg)
        with st.chat_message("user"):
            st.markdown(f"**{st.session_state.selected_user}**: {prompt}")
        if "@colaborai" in prompt.lower():
            if st.session_state.agent is None:
                with st.chat_message("assistant"):
                    st.warning("Create the agent in the sidebar before asking questions.")
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        result = st.session_state.agent.invoke({
                            "messages": [
                                {"type": "human", "content": prompt.replace("@colaborai", "").strip()}
                            ]
                        })
                        content = result["messages"][-1].content
                        st.markdown(content)
                        assistant_msg = {"role": "assistant", "content": content}
                        st.session_state.messages.append(assistant_msg)
                        append_history_to_file(assistant_msg)

    # st.divider()
    # st.subheader("Collaboration Simulation")
    # st.caption(
    #     "This MVP uses a shared Chroma persist directory (set via `RAG_VDB_DIR` or `./vdb`). "
    #     "Run this app in multiple browsers to simulate different users chatting over the same index."
    # )

if __name__ == "__main__":
    main()