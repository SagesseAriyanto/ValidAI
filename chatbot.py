from google import genai
from dotenv import load_dotenv
import pandas as pd
import os
import json
import streamlit as st
import numpy as np

load_dotenv()


@st.cache_data
def load_chatbot_data():
    df = pd.read_csv("./ai_data.csv")

    # Map redundant price values to 'Free'
    price_mapping = {"GitHub": "Free", "Open Source": "Free", "Google Colab": "Free"}
    df["Price"] = df["Price"].replace(price_mapping)
    df["Price"] = df["Price"].fillna("N/A")
    df.dropna(inplace=True)

    # Create unique lists from DataFrame columns
    categories_list = ", ".join(df["Category"].unique())
    prices_list = ", ".join(df["Price"].unique())

    return df, categories_list, prices_list

@st.cache_data
def get_available_models():
    models_file = "gemini_models.json"

    if os.path.exists(models_file):
        with open(models_file, "r") as file:
            models_list = json.load(file)
        return models_list
    models_list = []
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    for model in client.models.list():
        if "generateContent" in model.supported_actions:
            model_name = model.name.replace("models/", "")
            models_list.append(model_name)

    # Save models to a JSON file
    with open(models_file, "w") as file:
        json.dump(models_list, file)
    return models_list


@st.cache_resource
def get_genai_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_resp(chat_history: list) -> str:
    # Load data and initialize client
    df, categories_list, prices_list = load_chatbot_data()
    client = get_genai_client()
    models = get_available_models()

    if df.empty: return "Error: Database not loaded."
    if not chat_history: return "Hello! How can I help you?"

    # Extract the latest question and past history
    current_question = chat_history[-1]["content"]
    past_history = chat_history[:-1][-3:]

    history_text = ""
    for msg in past_history:
        role = "User" if msg["role"] == "user" else "AI"
        content = str(msg["content"])[:150]
        history_text += f"{role}: {content}\n"

    # Hybrid prompt with context
    prompt = f"""
    Pandas DataFrame 'df' of AI Tools.
    SCHEMA:
    - Name (str): Tool name
    - Upvotes (int): User popularity count (Higher is better)
    - Link (str): Website URL
    - Price (str): {prices_list}
    - Category (str): {categories_list}
    - Description (str): Summary of what the tool does

    HISTORY:
    {history_text}
    
    QUESTION: "{current_question}"

    INSTRUCTIONS:
    1. GENERAL CHAT: If user greets or asks concepts, reply with plain text.
    2. DATA QUERY: Write a SINGLE Python Expression.
       - SYNONYMS: If user asks for 'users', 'popularity', 'traffic', or 'best', use the 'Upvotes' column.
       - COLUMNS: Select the exact columns the user asked for.
         * Default if unspecified: `[['Name', 'Link', 'Upvotes']]`
       - FORMATTING: ALWAYS use `.head(n)` (even for 1 result) to return a DataFrame.
       - SAFETY: Return ONLY the raw code string. DO NOT use print().
    """
    response = None
    for model in models:
        try:
            response = client.models.generate_content(
                model=model, contents=prompt
                )
            break
        except:
            continue
    if not response:
        return f"System overloaded. Please try again later."

    # Clean generated code
    code = (
        response.text.replace("```python", "")
        .replace("```", "")
        .replace("`", "")
        .strip()
    )
    # Strip print()
    if code.startswith("print(") and code.endswith(")"):
        code = code[6:-1].strip()
    
    # Remove any trailing print statements
    if "print(" in code:
        code = code.split("print(")[0].strip()
   
    if code.startswith("df") or code.startswith("pd."):
        # If response is code, run it
        try:
            # Run the generated code
            result = eval(code)

            # DataFrame (Standard Table)
            if isinstance(result, pd.DataFrame):
                if result.empty:
                    return "No results found."
                return result.to_markdown(index=False)

            # Series (Single Column)
            elif isinstance(result, pd.Series):
                return result.to_frame().T.to_markdown(index=False)

            else:
                return str(result)

        except:
            return f"I couldn't process that query. Try rephrasing."
    else:
        return code

get_resp([{"role": "user", "content": "Hello"}])  # Warm up the cache
