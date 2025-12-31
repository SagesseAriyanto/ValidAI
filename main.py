import streamlit as st
import pickle
import numpy as np


def predict_category(desciption):
    category_model = pickle.load(open("./Models/model_category.pkl", "rb"))
    category_vectorizer = pickle.load(open("./Models/vectorizer_category.pkl", "rb"))
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
    category_encoder = pickle.load(open("./Models/label_encoder_category.pkl", "rb"))
    price_encoder = pickle.load(open("./Models/label_encoder_price.pkl", "rb"))
    # Vectorize the input description
    desc_vec = description_vectorizer.transform([description]).toarray()
    category_enc = category_encoder.transform([category])
    price_enc = price_encoder.transform([price])

    # Combine all features into a single feature set
    features = np.hstack(
        (desc_vec, category_enc.reshape(-1, 1), price_enc.reshape(-1, 1))
    )
    # Predict success
    success_prob = int(success_model.predict_proba(features)[0][1] * 100)  # probability of success
    return success_prob


st.title("AI Predictor")

company_name = st.text_input("Enter your AI company name:")
description = st.text_area("Enter a brief description of your AI company:")

if st.button("Predict"):
    if description:
        category = predict_category(description)
        st.success(f"Category: {category} ✔️")
        st.subheader("Success Probability by Price")
        price_types = ["Free", "Freemium", "Paid"]
        for price in price_types:
            success_prob = predict_success(description, category, price)
            st.write(f"{price}: {success_prob}% chance of success")
