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

st.title("AI Predictor")

company_name = st.text_input("Enter your AI company name:")
description = st.text_area("Enter a brief description of your AI company:")

if st.button("Predict Category"):
    if description:
        category = predict_category(description)
        st.success(f"The predicted category for your AI company is: {category}")
    else:
        st.error("Please enter a description to get a prediction.")