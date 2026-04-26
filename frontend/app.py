import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="AI Shopping Assistant")

st.title("🛒 AI Shopping Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask something about products...")

if user_input:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Send to backend
    response = requests.post(
        API_URL,
        json={"messages": st.session_state.messages}
    )

    answer = response.json()["answer"]

    # Add assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    # Show assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)