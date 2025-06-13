from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_helper import chat_with_memory

app = FastAPI()

# Pydantic model for POST request body
class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat/")
async def chat_endpoint(request: ChatRequest):
    """POST endpoint to send user input to Langchain and get response."""
    user_text = request.user_input
    response = chat_with_memory(user_text)
    return {"response": response}
