# ============================================
# IMPORT LIBRARIES
# ============================================
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import openai, langchain_openai, streamlit as st



# ============================================
# PAGE CONFIGURATION - MUST BE FIRST!
# ============================================
st.set_page_config(
    page_title="TechCorp AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.write("openai:", openai.__version__)
# st.write("langchain-openai:", langchain_openai.__version__)

# ============================================
# LOAD API KEY - WITH DEBUGGING
# ============================================
api_key = None

# Try st.secrets first (for Streamlit Cloud)
try:
    api_key = st.secrets["OPENAI_API_KEY"].strip()  # Strip whitespace!
    st.success("✅ API key loaded from Streamlit secrets")
except Exception as e:
    st.warning(f"⚠️ Could not load from secrets: {str(e)}")
    # Try .env file (for local)
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        api_key = api_key.strip()  # Strip whitespace!
        st.success("✅ API key loaded from .env file")

# Verify API key exists and is valid
if not api_key:
    st.error("❌ OPENAI_API_KEY not found!")
    st.info("📝 Please add your OpenAI API key to:")
    st.code("""
# In Streamlit Cloud: Settings → Secrets
OPENAI_API_KEY = "sk-proj-..."

# Or locally: Create .env file
OPENAI_API_KEY=sk-proj-...
    """)
    st.stop()

if not api_key.startswith("sk-"):
    st.error("❌ Invalid API key format! Should start with 'sk-'")
    st.stop()

# Set environment variable for LangChain
os.environ["OPENAI_API_KEY"] = api_key

# ============================================
# DISPLAY TITLE AND DESCRIPTION
# ============================================
st.title("🤖 TechCorp Assistant")
st.caption("Ask me anything about TechCorp!")

# ============================================
# INITIALIZE SESSION STATE
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    
    # ----------------------------------------
    # STEP 1: LOAD VECTOR DATABASE
    # ----------------------------------------
    with st.spinner("Loading vector database..."):
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            
            vectorstore = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings
            )
            st.success("✅ Vector database loaded")
        except Exception as e:
            st.error(f"❌ Error loading vector database: {str(e)}")
            st.stop()
    
    # ----------------------------------------
    # STEP 2: CREATE MEMORY
    # ----------------------------------------
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # ----------------------------------------
    # STEP 3: CREATE THE AI CHAIN
    # ----------------------------------------
    try:
        st.session_state.chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                model_kwargs={
                    "top_p": 1,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                }
            ),
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 2}
            ),
            memory=memory,
            return_source_documents=True
        )
        st.success("✅ AI chain initialized")
    except Exception as e:
        st.error(f"❌ Error creating AI chain: {str(e)}")
        st.stop()

# ============================================
# DISPLAY CHAT HISTORY
# ============================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ============================================
# CHAT INPUT BOX
# ============================================
if prompt := st.chat_input("Type your question here..."):
    
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("💭"):
            response = st.session_state.chain.invoke({"question": prompt})
            answer = response["answer"]
            st.write(answer)
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })