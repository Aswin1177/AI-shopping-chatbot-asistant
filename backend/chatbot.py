import data

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
    data_store.append({"text":row["combined_texts"],
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
genai.configure(api_key="AIzaSyBaXIbC6Ax3821uzhfxGQiBu6DwATKo2O0")

model=genai.GenerativeModel('gemini-2.5-flash')

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

    # Format context cleanly
    formatted_context = ""
    for item in context:
        formatted_context += f"""
        Product:
        {item['text']}
        Price: ₹{item['discounted_price']}
        Original Price: ₹{item['actual_price']}
        Discount: {item['discount_percentage']}%
        Rating: {item['rating']} ({item['rating_count']} reviews)
        """

            # Build prompt
    prompt = f"""
    You are an AI shopping assistant.

    STRICT RULES:
    - Use ONLY the provided context
    - Do NOT hallucinate
    - If answer is not found, say: "Sorry, I cant help you with that"
    - Keep answers brief (under 500 words)

    CONTEXT:
    {formatted_context}

    USER QUESTION:
    {query}
        """


    response = chat.send_message(
        prompt,
        generation_config={"temperature": 0.2}
    )

    return response.text