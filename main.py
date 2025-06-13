from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_helper import chat_with_memory  # Your custom function

app = FastAPI()

# ✅ CORS Middleware integration here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ In production, change this to the exact domain (e.g., ["https://your-frontend.com"])
    allow_credentials=True,
    allow_methods=["*"],   # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],   # Allow all headers
)

# ✅ Request Body model
class ChatRequest(BaseModel):
    user_input: str

# ✅ POST endpoint
@app.post("/chat/")
async def chat_endpoint(request: ChatRequest):
    """POST endpoint to send user input to Langchain and get response."""
    user_text = request.user_input
    response = chat_with_memory(user_text)
    return {"response": response}
