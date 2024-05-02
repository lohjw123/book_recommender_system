import streamlit as st
from joblib import load
import numpy as np
import pandas as pd


# Load the model
model = load('book_recommender_model.joblib')

# Function to recommend books
def recommend(title, model=model):
    return model[title]

# Streamlit UI
st.title("Book Recommender System")

# Select book
selected_book = st.selectbox("Select a book: ")

if st.button('Recommend'):
    recommendations = recommend(selected_book)
    st.write("## Recommended Books:")
    for book in recommendations:
        st.write("- " + book)