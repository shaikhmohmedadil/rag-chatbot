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

st.set_page_config(
    page_title="Interkultureller Garten Coswig Assistant",
    page_icon="🌱",
    layout="centered"
)

# ============================================
# LOAD API KEY
# ============================================
# Try Streamlit Cloud secrets first, then .env file
if "OPENAI_API_KEY" not in os.environ:
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except:
        load_dotenv()


# ============================================
# DISPLAY TITLE
# ============================================
st.title("🌿 Interkultureller Garten Coswig Assistant")
st.caption("Frag mich alles über den Interkulturellen Garten Coswig e.V.")

# ============================================
# INITIALIZE SESSION STATE
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    try:
        # Load vector database
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        
        # Create conversation memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create AI chain
        st.session_state.chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0
            ),
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
            memory=memory,
            return_source_documents=True
        )
    except Exception as e:
        st.error(f"❌ Error initializing chatbot: {e}")
        st.stop()

# ============================================
# DISPLAY CHAT HISTORY
# ============================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ============================================
# CHAT INPUT
# ============================================
if prompt := st.chat_input("Type your question here..."):
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("💭"):
            try:
                response = st.session_state.chain.invoke({"question": prompt})
                answer = response["answer"]
                st.write(answer)
                
                # Add to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {e}")