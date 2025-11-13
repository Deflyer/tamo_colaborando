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

def build_or_update_index(uploaded_pdf_bytes: bytes, filename: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
        tmp.write(uploaded_pdf_bytes)
        tmp_path = tmp.name

    embeddings = build_embeddings()
    pages = load_pdf_pages(tmp_path)
    vectorstore = build_vectorstore_from_pages(
        pages,
        embeddings,
        persist_directory=get_shared_vectorstore_dir(),
        collection_name="book",
    )
    retriever = build_retriever(vectorstore)
    os.unlink(tmp_path)
    return retriever


def main():
    load_dotenv()
    st.set_page_config(page_title="RAG Chat (aula)", page_icon="📄")
    ensure_session_state()

    st.title("📄 RAG Chat (aula)")
    st.caption("Upload a PDF, then chat. Multiple users share the same vector store.")

    with st.sidebar:

        st.header("User")
        st.session_state.selected_user = st.selectbox("Active user", USERS, index=USERS.index(st.session_state.selected_user))
        st.caption("Messages sent will be attributed to this user.")

        st.header("Data")
        # upload PDF
        uploaded = st.file_uploader("Upload PDF", type=["pdf"])  # pronto no streamlit para carregar arqruivos
        # if success, allow button to build/update vectorDB index
        if uploaded is not None:
            if st.button("Build/Update Index", type="primary"):
                with st.spinner("Creating index... "):
                    st.session_state.retriever = build_or_update_index(uploaded.read(), uploaded.name)
                st.success("Index created.")
        
        st.divider()
        st.header("Agent")
        # slider k
        k = st.slider("Retrieve k chunks", min_value=2, max_value=10, value=6, step=2)

        # slider temperature
        temp = st.slider("Model temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)

        # button for agent creation
        if st.button("(re)Create Agent"):
            if st.session_state.retriever is None:
                st.warning("Please build the index first")
            else:
                llm = build_llm(temperature=temp)
                retriever = st.session_state.retriever
                retriever.search_kwargs["k"] = k
                st.session_state.agent = build_agent(retriever, llm)
                st.success("Agent ready.")

    # Chat area
    st.subheader("Espa;o para discussão entre os usuários, caso desej chamar o agente, marque o com @colaborai na mensagem")
    for msg in st.session_state.messages:
        author = msg.get("user", "User") if msg["role"] == "user" else "assistant"
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(f"**{author}**: {msg['content']}")

    if prompt := st.chat_input(f"{st.session_state.selected_user} diz: "):
        st.session_state.messages.append({"user": st.session_state.selected_user, "role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f"**{st.session_state.selected_user}**: {prompt}")
        if "@colaborai" in prompt.lower():
            if st.session_state.agent is None:
                with st.chat_message("assistant"):
                    st.warning("Create the agent in the sidebar before asking questions.")
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Use LC message dict format expected by the agent
                        result = st.session_state.agent.invoke({
                            "messages": [
                                {"type": "human", "content": prompt.replace("@colaborai", "").strip()}
                            ]
                        })
                        content = result["messages"][-1].content
                        st.markdown(content)
                        st.session_state.messages.append({"role": "assistant", "content": content})

    # st.divider()
    # st.subheader("Collaboration Simulation")
    # st.caption(
    #     "This MVP uses a shared Chroma persist directory (set via `RAG_VDB_DIR` or `./vdb`). "
    #     "Run this app in multiple browsers to simulate different users chatting over the same index."
    # )

if __name__ == "__main__":
    main()