from fastapi import FastAPI, Header, HTTPException
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import List
import chatbot

load_dotenv()

app = FastAPI()

SECRET = os.getenv("APP_SECRET")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


@app.post("/chat")
def chat(data: ChatRequest, x_api_key: str = Header(None)):

    if x_api_key != SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")

    if not data.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    user_query = data.messages[-1].content.strip()

    if not user_query:
        raise HTTPException(status_code=400, detail="Empty message")

    try:
        answer = chatbot.chatbot_response(user_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"response": answer}