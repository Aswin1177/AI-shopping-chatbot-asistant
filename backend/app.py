from fastapi import FastAPI, Header, HTTPException
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import chatbot

load_dotenv()

app=FastAPI()

SECRET = os.getenv("APP_SECRET")

from typing import List

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.post("/chat")

def chat(data: dict, x_api_key: str = Header(None)):

    if x_api_key != SECRET:

        raise HTTPException(status_code=403, detail="Unauthorized")

    user_query = data["messages"][0]["content"]

    answer = chatbot.chatbot_response(user_query)

    return {"response": answer}