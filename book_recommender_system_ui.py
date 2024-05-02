import streamlit as st
from joblib import load
import numpy as np
import pandas as pd


# Load the model
model = load('book_recommender_model.joblib')

# Load the pre-trained model
model = joblib.load('book_recommender_system.joblib')

# Function to recommend books
def recommend_books(title, model=model):
    return model[title]

# Streamlit UI
st.title('Book Recommender System')

# Select book
selected_book = st.selectbox('Select a book:', books_df['title'].values)

if st.button('Recommend'):
    recommendations = recommend_books(selected_book)
    st.write('## Recommended Books:')
    for book in recommendations:
        st.write('- ' + book)