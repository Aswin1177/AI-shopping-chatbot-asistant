import streamlit as st
import requests

API_URL = "http://13.201.20.19:8000/chat"
API_SECRET = "password"

st.set_page_config(page_title="Demo-Amazon Shopping Assistant Project")

st.title("😼 Aswin's Shopping Assistant chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask something about products...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    with st.chat_message("user"):
        st.markdown(user_input)
    try:
        response = requests.post(
            API_URL,
            headers={"x-api-key": API_SECRET},
            json={"messages": st.session_state.messages}
        )
        answer = response.json().get("response", response.text)
    except Exception as e:
        answer = f"Request failed: {str(e)}"
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.markdown(answer)