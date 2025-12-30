import streamlit as st
import pickle

def predict_category(desciption):
    category_model = pickle.load(open("./Models/model_category.pkl", "rb"))
    category_vectorizer = pickle.load(
        open("./Models/vectorizer_category.pkl", "rb")
    )
    # Vectorize the input description (convert to numerical data)
    desc_vec = category_vectorizer.transform([desciption])
    # Predict the category
    category_prediction = category_model.predict(desc_vec)[0]
    return category_prediction

def predict_success(description, category, price):
    success_model = pickle.load(open("./Models/model_success.pkl", "rb"))
    description_vectorizer = pickle.load(
        open("./Models/vectorizer_description.pkl", "rb")
    )
    category_encoder = pickle.load(
        open("./Models/label_encoder_category.pkl", "rb")
    )
    price_encoder = pickle.load(
        open("./Models/label_encoder_price.pkl", "rb")
    )
    

st.title("AI Predictor")

company_name = st.text_input("Enter your AI company name:")
description = st.text_area("Enter a brief description of your AI company:")

if st.button("Predict"):
    if description:
        category = predict_category(description)
        st.success(f"The predicted category for your AI company is: {category}")
    else:
        st.error("Please enter a description to get a prediction.")