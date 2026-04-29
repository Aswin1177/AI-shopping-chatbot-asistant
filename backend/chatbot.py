import data
import os
from dotenv import load_dotenv
load_dotenv()
def combine_texts(row):
    return f"""Product_name :{row["product_name"]},Category:{row["category"]},
    About:{row["about_product"]}, Review title: {row["review_title"]},
    Review: {row["review_content"]}"""

data.df["combined_texts"]=data.df.apply(combine_texts,axis=1)
from sentence_transformers import SentenceTransformer
txt_model=SentenceTransformer('all-MiniLM-L6-V2')

texts=data.df["combined_texts"].tolist()
embeddings=txt_model.encode(texts).astype('float32')

import faiss
dims=embeddings.shape[1]
index=faiss.IndexFlatL2(dims)
index.add(embeddings)

data_store=[]
for _,row in data.df.iterrows():
    data_store.append({"product_name":row["product_name"],
                       "category": row["category"],
                       "about": row["about_product"],
                       "review_title": row["review_title"],
                       "review": row["review_content"],
                       "discounted_price":row["discounted_price"],
                       "actual_price":row["actual_price"],
                       "discount_percentage": row["discount_percentage"],
                       "rating":row["rating"],
                       "rating_count":row["rating_count"]})

def retrieve(query, k=2):
    query_emb = txt_model.encode([query]).astype('float32')
    distances, indices = index.search(query_emb, k)

    results = []
    for i in indices[0]:
        item = data_store[i]
        results.append(item)

    return results

 
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model=genai.GenerativeModel('gemini-2.5-flash-lite')

chat=model.start_chat(history=[])
last_queries=[]
def chatbot_response(query):
    global last_queries
    # Exit condition
    if query.lower() in ["exit", "quit", "stop"]:
        return "Session ended."
    last_queries.append(query)
    last_queries=last_queries[-3:]
    refined_query=" ".join(last_queries)
    # Retrieve relevant context
    context = retrieve(refined_query, k=3)

    def get_index_from_query(query):
        q = query.lower()
        if "first" in q or "1" in q:
            return 0
        elif "second" in q or "2" in q:
            return 1
        elif "third" in q or "3" in q:
            return 2
        return None
    global last_results


    if query.lower() in ["exit", "quit", "stop"]:
        return "Session ended."

    idx = get_index_from_query(query)

    if idx is not None and len(last_results) > idx:
        item = last_results[idx]

        return f"""
        Product: {item['product_name']}
        Price: ₹{item['discounted_price']}
        Original Price: ₹{item['actual_price']}
        Discount: {item['discount_percentage']}%
        Rating: {item['rating']} ({item['rating_count']} reviews)
        """

    # -------- STEP 2: RETRIEVE (ONLY CURRENT QUERY) --------
    last_results = context 
    formatted_context = ""
    for item in context:
        formatted_context += f"""
        Product Name: {item['product_name']}
        Category: {item['category']}
        About:{item['about']}
        Review Title:{item['review_title']}
        Full Review:{item['review']}
        Price: ₹{item['discounted_price']}
        Original Price: ₹{item['actual_price']}
        Discount: {item['discount_percentage']}%
        Rating: {item['rating']} ({item['rating_count']} reviews)
        """
    last_results = []
    
    global history_text
    history_text=" "

            # Build prompt
    prompt = f"""
    You are an AI Shopping Chatbot.

    Your task is to respond intelligently based on the USER QUERY using the provided CONTEXT.

    CONTEXT FORMAT (IMPORTANT)

    Each product in CONTEXT is described as:

    Product Name: <product_name>
    Category: <category>
    About: <about_product>
    Review Title: <review_title>
    Full Review: <review_content>
    Price: ₹<discounted_price>
    Original Price: ₹<actual_price>
    Discount: <discount_percentage>%
    Rating: <rating> (<rating_count> reviews)

    IMPORTANT:
    - "About" contains product features and specifications
    - Use reviews only as supporting evidence
    - Focus mainly on product features, price, and rating

    RULES (STRICT)

    1. Use ONLY the products in CONTEXT
    2. Do NOT hallucinate
    3. If no products are not found, say:
    "Sorry, I can't help you with that"
    4. Keep your responses brief (limit 0 to 600 words)
    5. If the user is:
        - greeting (hi, hello)
        - casual talk (ok, nice, cool)
        - appreciation (thanks)
        - closing (bye)
    → Use only the relevant information from CONTEXT to keep the conversation going

    → Respond naturally like a human  

    --- BEHAVIOR CONTROL (CRITICAL) ---

    A. If user asks for recommendations → suggest EXACTLY 3 products  
    B. If asking about a product → answer ONLY that, no new suggestions  
    C. If user gives feedback (ok, nice, good, etc.) → respond conversationally  
    D. If unclear → ask clarification  

    --- CONTEXT ---
    {formatted_context}

    --- PREVIOUS PRODUCTS SHOWN ---
    {[item['product_name'] for item in last_results]}

    --- USER QUERY ---
    {query}

    RESPONSE FORMAT:

    If recommending:

    Top Recommendations:

    1. Product: <name>
    - Price: ₹<discounted_price> (Original: ₹<actual_price>)
    - Discount: <discount_percentage>%
    - Rating: <rating> (<rating_count> reviews)
    - Reason: <why it matches>

    (3 items total)

    Otherwise:
    - Respond naturally
    - Use CONVERSATION HISTORY to identify if the query is connected to previous chat and respond accordingly
    - DO NOT force recommendations
    CONVERSATION HISTORY:
    {history_text}

    FINAL INSTRUCTION:
    Be concise, human-like, and useful.
    """

        # -------- STEP 5: GENERATE RESPONSE --------
    response = chat.send_message(
            prompt,
            generation_config={"temperature": 0.2}
    )

    for msg in chat.history:
        if hasattr(msg, "role"):  # Content object
            role = msg.role
            content = msg.parts[0].text if msg.parts else ""
        else:  # dict
            role = msg.get("role", "")
            parts = msg.get("parts", [])
            content = parts[0].get("text", "") if parts else ""

        history_text += f"{role.capitalize()}: {content}\n"

    return response.text