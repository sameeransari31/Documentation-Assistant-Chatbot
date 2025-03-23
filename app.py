import streamlit as st
import uuid
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import os
from dotenv import load_dotenv
load_dotenv()

store = {}

def get_session_id() -> str:
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

st.set_page_config(
    page_title="DocuChat - Smart Documentation Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stChatInput {position: fixed; bottom: 20px; width: 60%;}
    .reportview-container .main .block-container {padding-bottom: 150px;}
    .sidebar .sidebar-content {background-color: #f0f2f6;}
    [data-testid="stHeader"] {background-color: rgba(0,0,0,0);}
    .session-info {color: #666; font-size: 0.8em; margin-top: 10px;}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

with st.sidebar:
    st.title("⚙️ Settings")
    groq_api_key = st.text_input("Enter Groq API Key:", type="password")
    model_name = st.selectbox(
        "Choose Model:",
        ["Gemma2-9b-It", "Llama3-8b-8192", "Mixtral-8x7b-32768"]
    )
    temperature = st.slider("Temperature:", 0.0, 1.0, 0.3)
    max_tokens = st.slider("Max Tokens:", 128, 4096, 1024)
    st.markdown(f"<div class='session-info'>Session ID: {get_session_id()}</div>", 
                unsafe_allow_html=True)

st.title("📚 DocuChat - Documentation Assistant")
st.markdown("""
Ask questions about:
- Pydantic
- Mistral
- React
- Next.js
- TailwindCSS
- LangChain
""")

@st.cache_resource(show_spinner=False)
def load_documents():
    with st.status("📥 Loading documentation...", expanded=True) as status:
        urls = [
            "https://docs.pydantic.dev/latest/",
            "https://docs.mistral.ai/",
            "https://react.dev/",
            "https://nextjs.org/docs",
            "https://tailwindcss.com/docs",
            "https://python.langchain.com/docs/",
        ]
        st.write("Fetching documents from websites...")
        loader = WebBaseLoader(urls)
        documents = loader.load()
        st.write("Processing documents...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(documents)
        st.write("Creating knowledge base...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        status.update(label="✅ Documents loaded!", state="complete")
        return vectorstore

if st.session_state.vectorstore is None:
    st.session_state.vectorstore = load_documents()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about documentation..."):
    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar!")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("🤔 Thinking..."):
        try:
            model = ChatGroq(
                model=model_name,
                groq_api_key=groq_api_key,
                temperature=temperature,
                max_tokens=max_tokens
            )
            retriever = st.session_state.vectorstore.as_retriever()
            
            def get_chat_history(session_id: str) -> ChatMessageHistory:
                if session_id not in store:
                    store[session_id] = ChatMessageHistory()
                return store[session_id]
            
            contextualize_q_system_prompt = """Given a chat history and the latest user question, 
            formulate a standalone question. Return it as is if already standalone."""
            
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            
            que_ans_prompt = ChatPromptTemplate.from_messages([
                ("system", """Answer concisely using context. If unsure, say so. Max 3 sentences.
                Context: {context}"""),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}")
            ])
            
            history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)
            qa_chain = create_stuff_documents_chain(model, que_ans_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
            
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_chat_history,
                input_messages_key="input",
                output_messages_key="answer",
                history_messages_key="chat_history"
            )
            
            response = conversational_rag_chain.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": get_session_id()}}
            )['answer']
            
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

col1, col2 = st.columns([0.8, 0.2])
with col2:
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        session_id = get_session_id()
        if session_id in store:
            del store[session_id]
        st.experimental_rerun()

with st.expander("📚 Supported Documentation Sources"):
    st.markdown("""
    - **Pydantic**: Python data validation and settings management
    - **Mistral**: AI model documentation
    - **React**: JavaScript library for building user interfaces
    - **Next.js**: React framework for production
    - **TailwindCSS**: Utility-first CSS framework
    - **LangChain**: Framework for developing AI applications
    """)
