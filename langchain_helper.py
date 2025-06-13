import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Latest free model - faster and more efficient
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7,
        max_tokens=None,  # Use default limits
        max_retries=2,
    )


# Memory for chat (stores past conversation)
memory = ConversationBufferMemory()

# Conversation chain with memory
conversation = ConversationChain(
    llm=llm,
    memory=memory
)

def chat_with_memory(user_input: str) -> str:
    """Process user input and return LLM output with memory tracking."""
    response = conversation.run(user_input)
    return response
