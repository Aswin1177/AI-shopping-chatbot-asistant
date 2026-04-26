from fastapi import FastAPI
from pydantic import BaseModel
import chatbot

app=FastAPI()

from typing import List

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.post("/chat")

def chat(req: ChatRequest):
    user_query = req.messages[-1].content
    answer = chatbot.chatbot_response(user_query)
    return {"answer": answer}
