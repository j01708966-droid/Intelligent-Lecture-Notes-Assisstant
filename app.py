import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import tempfile

st.set_page_config(page_title="Intelligent Lecture Notes Assistant")
st.title("📚 Intelligent Lecture Notes Assistant")

# ── Session state init ──────────────────────────────────────────────
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Sidebar: setup ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Setup")
    groq_api_key = st.text_input("Groq API Key", type="password").strip()
    uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    chunk_size = st.selectbox("Chunk size", [500, 1000, 1500], index=1)
    k = st.selectbox("Chunks to retrieve (k)", [3, 5, 8], index=1)
    build_btn = st.button("Build Knowledge Base")

# ── Build ONE Master Chain on button click ──────────────────────────
if build_btn:
    if not groq_api_key or not uploaded_files:
        st.sidebar.error("Please provide API key and at least one PDF.")
    else:
        with st.spinner("Loading PDFs and building vector store..."):
            os.environ["GROQ_API_KEY"] = groq_api_key
            documents = []
            for uf in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name
                loader = PyPDFLoader(tmp_path)
                documents.extend(loader.load())

            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
            docs = splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})

            llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

            def format_docs(docs):
                return "\n\n".join(d.page_content for d in docs)

            # ONE Master Prompt to handle everything
            master_prompt = ChatPromptTemplate.from_template("""
You are an Intelligent Lecture Notes Assistant. 
Use the following retrieved context from the lecture notes to fulfill the user's request. 

Guidelines:
- If the user asks for a summary, provide a clear and concise summary of the context.
- If the user asks for MCQs, quizzes, viva questions, or flashcards, generate them logically based on the context.
- If the user asks a specific question, answer it using *only* the provided context. 
- If you cannot find the answer in the context, say: "Sorry, I couldn't find this in the lecture notes."

Context: {context}

User Request: {question}
Answer:""")

            st.session_state.chain = (
                {"context": retriever | format_docs, "question": lambda x: x}
                | master_prompt
                | llm
                | StrOutputParser()
            )
            st.session_state.chat_history = []
        st.sidebar.success(f"✅ Loaded {len(documents)} pages, {len(docs)} chunks.")

# ── Pure Chat UI ─────────────────────────────────────────────────────
if st.session_state.chain:
    # Render history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Single chat input handles everything
    user_input = st.chat_input("Ask a question, or tell me to summarize, create MCQs, etc...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chain.invoke(user_input)
                st.markdown(result)
        st.session_state.chat_history.append({"role": "assistant", "content": result})

else:
    st.info("👈 Upload PDFs and click **Build Knowledge Base** to get started.")
