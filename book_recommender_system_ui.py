import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader

# Load the model
model = load('book_recommender_model.joblib')

# Load book data
book_data = pd.read_csv('Books.csv')

# Function to recommend books
def recommend(title, model=model):
    book_indices = book_data['Book-Title'].apply(lambda x: x.lower() == title.lower())
    book_index = book_indices[book_indices].index[0]

    # Get the book vector from the SVD components
    book_vector = model.components_.T @ model.transform([book_data.iloc[book_index].values])[0]

    # Calculate cosine similarities with all book vectors
    book_vectors = model.components_.T @ model.transform(book_data.values)
    similarities = [1 - cosine(book_vector, vector) for vector in book_vectors]

    # Sort similarities and get the top recommendations
    top_indices = np.argsort(-np.array(similarities))[1:6]
    top_books = book_data.iloc[top_indices]['Book-Title', 'Image-URL-L'].values

    return top_books

# Streamlit UI
st.title("Book Recommender System")

# Get unique book titles
book_titles = book_data['Book-Title'].unique()

# Select book
selected_book = st.selectbox("Select a book:", book_titles)

if st.button('Recommend'):
    recommendation = recommend(selected_book)
    st.write("## Recommended Book:")
    st.write("- " + str(recommendation))