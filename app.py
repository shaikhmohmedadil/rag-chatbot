# ============================================
# IMPORT LIBRARIES
# ============================================
# Think: Getting all the tools we need before starting

import streamlit as st  # Creates the chat UI (the website interface)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # Talks to OpenAI (GPT + Embeddings)
from langchain_community.vectorstores import Chroma  # Loads our vector database (chroma_db folder)
from langchain.chains import ConversationalRetrievalChain  # Handles Q&A with memory
from langchain.memory import ConversationBufferMemory  # Remembers conversation history
# from langchain.prompts import PromptTemplate  
from dotenv import load_dotenv  # Loads API key from .env file
import os


# ============================================
# PAGE CONFIGURATION
# ============================================
# Sets up how the webpage looks (title, icon, layout)
st.set_page_config(
    page_title="TechCorp AI Assistant",  # Browser tab title
    page_icon="🤖",  # Browser tab icon
    layout="centered"  # Chat appears in center (not full width)
)

try:
    # Streamlit Cloud
    api_key = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = api_key
except (FileNotFoundError, KeyError):
    # Local development - load from .env
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

# Verify API key exists
if not api_key or api_key == "":
    st.error("❌ OPENAI_API_KEY not found! Please add it to Streamlit Secrets or .env file")
    st.stop()

# ============================================
# LOAD API KEY
# ============================================
# load_dotenv()  # Reads .env file and loads OPENAI_API_KEY


# ============================================
# DISPLAY TITLE AND DESCRIPTION
# ============================================
st.title("🤖 TechCorp Assistant")  # Big heading at top
st.caption("Ask me anything about TechCorp!")  # Small text under heading

# Real-life analogy:
# Like putting a sign above your shop: "Welcome to TechCorp Support!"

# ============================================
# INITIALIZE SESSION STATE
# ============================================
# Session state = Memory that persists as user chats
# Like: Computer's short-term memory during this conversation

# Check if "messages" exists in memory, if not, create empty list
if "messages" not in st.session_state:
    st.session_state.messages = []
    # This will store: [
    #   {"role": "user", "content": "What services?"},
    #   {"role": "assistant", "content": "We offer AI consulting..."}
    # ]

# Check if "chain" (the AI brain) exists, if not, create it
if "chain" not in st.session_state:
    
    # ----------------------------------------
    # STEP 1: LOAD VECTOR DATABASE
    # ----------------------------------------
    print("Loading vector database...")  # Shows in terminal (not visible to user)
    
    # Create embeddings tool (converts text to vectors)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    # Load the chroma_db folder you created with ingest.py
    vectorstore = Chroma(
        persist_directory="./chroma_db",  # Where ingest.py saved data
        embedding_function=embeddings  # How to search (using vectors)
    )
    
    # Real-life analogy:
    # Opening the library (chroma_db) so we can search through books
    
    # ----------------------------------------
    # STEP 2: CREATE MEMORY
    # ----------------------------------------
    # Memory = Remembers what user said earlier in conversation
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # Name for storing history
        return_messages=True,  # Return as message format
        output_key="answer"  # Where to find bot's answer
    )
    
    # Real-life analogy:
    # Like a notebook where you write down conversation:
    # User: "What services?"
    # You: "AI consulting, chatbots..."
    # User: "How much?" ← You remember they asked about services before
    
    # ----------------------------------------
    # STEP 3: CREATE THE AI CHAIN
    # ----------------------------------------
    # Chain = Connects everything together:
    # Vector DB → GPT → Answer
    
    st.session_state.chain = ConversationalRetrievalChain.from_llm(
        # The "brain" - GPT model that generates answers
        llm=ChatOpenAI(
            model="gpt-3.5-turbo",  # Which GPT model to use
            temperature=0,  # 0 = Precise answers, 1 = Creative answers
            model_kwargs={
                "top_p": 1, #Prevents weird “cut-off” answers
                "frequency_penalty": 0, #Allows repeating important terms
                "presence_penalty": 0 #Prevents the model from forcing new topics
            }
        ),

        # The "memory" - Where to search for info
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 2}  # Find top 3 most relevant chunks
        ),
        
        # The "notebook" - Conversation memory
        memory=memory,
        
        # Return the source chunks (optional, for debugging)
        return_source_documents=True
    )
    
    # Real-life analogy of the chain:
    # User asks question
    #   ↓
    # Search library (vectorstore) for relevant books (top 3 chunks)
    #   ↓
    # Read those books + remember previous conversation (memory)
    #   ↓
    # GPT writes answer based on books + context
    #   ↓
    # Show answer to user

# ============================================
# DISPLAY CHAT HISTORY
# ============================================
# Show all previous messages in the conversation

# Loop through each message stored in memory
for message in st.session_state.messages:
    # Display message bubble
    with st.chat_message(message["role"]):  # "user" or "assistant"
        st.write(message["content"])  # The actual text

# Real-life analogy:
# Like scrolling up in WhatsApp to see previous messages

# Example of what this displays:
# 👤 User: "What services do you offer?"
# 🤖 Bot: "We offer AI consulting, chatbot development, and data analysis"
# 👤 User: "How much?"
# 🤖 Bot: "Basic plan is ₹10,000/month..."

# ============================================
# CHAT INPUT BOX
# ============================================
# The text box at bottom where user types

# st.chat_input creates the input box
# := is "walrus operator" - assigns AND checks in one line
if prompt := st.chat_input("Type your question here..."):
    
    # User typed something! Let's process it.
    
    # ----------------------------------------
    # STEP 1: ADD USER MESSAGE TO HISTORY
    # ----------------------------------------
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Display user's message immediately
    with st.chat_message("user"):
        st.write(prompt)
    
    # Real-life analogy:
    # User says: "What are your office hours?"
    # You write it down in notebook (messages list)
    # You show it on screen (st.write)
    
    # ----------------------------------------
    # STEP 2: GET AI RESPONSE
    # ----------------------------------------
    
    # Display "thinking" animation while AI processes
    with st.chat_message("assistant"):
        with st.spinner("💭"):  # Shows spinning animation
            
            # THIS IS THE MAGIC! ✨
            # Send question to the chain we created earlier
            #response = st.session_state.chain({"question": prompt})
            response = st.session_state.chain.invoke({"question": prompt})

            
            # What happens inside chain:
            # 1. Takes user question: "What are your office hours?"
            # 2. Converts to vector: [0.23, 0.87, ...]
            # 3. Searches chroma_db for similar vectors
            # 4. Finds relevant chunks:
            #    Chunk: "Contact: ... Office Hours: Monday to Friday, 9 AM to 6 PM"
            # 5. Sends to GPT with context:
            #    "Based on this info: [chunk], answer: What are your office hours?"
            # 6. GPT generates natural answer:
            #    "Our office is open Monday to Friday from 9 AM to 6 PM!"
            # 7. Returns answer
            
            # Extract the answer from response
            answer = response["answer"]
            
            # Display the answer
            st.write(answer)
    
    # Real-life analogy:
    # User asks question → You search your files → Find answer → Tell user
    
    # ----------------------------------------
    # STEP 3: ADD BOT RESPONSE TO HISTORY
    # ----------------------------------------
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
    
    # Save bot's answer to memory so:
    # 1. It shows when page refreshes
    # 2. Bot remembers what it said (for follow-up questions)

# ============================================
# HOW THE FULL FLOW WORKS (Example)
# ============================================

# Example conversation:

# Turn 1:
# User types: "What services do you offer?"
#   ↓
# Chain searches chroma_db:
#   Finds chunk: "Services: - AI Consulting - Custom Chatbot Development - Data Analysis"
#   ↓
# Sends to GPT: "Based on [that chunk], answer: What services do you offer?"
#   ↓
# GPT responds: "We offer three main services: AI consulting, custom chatbot 
#                development, and data analysis. Which would you like to know more about?"
#   ↓
# Displays to user

# Turn 2:
# User types: "Tell me about the first one"
#   ↓
# Memory remembers: User asked about services, bot listed 3, "first one" = AI consulting
#   ↓
# Chain searches: Finds info about AI consulting
#   ↓
# GPT responds with context: "AI Consulting helps you identify where AI can improve 
#                             your business. We analyze your processes..."
#   ↓
# Displays to user

# This is the POWER of ConversationalRetrievalChain:
# - Searches documents (Retrieval)
# - Remembers conversation (Conversational)
# - Connects everything (Chain)

# ============================================
# END OF CODE
# ============================================
