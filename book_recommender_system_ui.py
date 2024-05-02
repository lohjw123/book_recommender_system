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
def recommend(title, model=model, book_data=book_data):
    # Convert book data to a surprise dataset
    reader = Reader(rating_scale=(1, 5))
    book_dataset = Dataset.load_from_df(book_data[['Book-Title']], reader)

    # Find the index of the input book
    book_indices = book_data['Book-Title'].apply(lambda x: x.lower() == title.lower())
    book_index = book_indices[book_indices].index[0]

    # Get the book's inner id from the dataset
    book_inner_id = book_dataset.to_inner_iid(book_data.iloc[book_index]['Book-Title'])

    # Get recommendations from the model
    recommendations = model.get_top_n(book_inner_id, n=5)

    # Convert recommendations to book titles and image URLs
    top_books = [(book_data[book_data['Book-Title'] == book_dataset.to_raw_iid(rec[0])]['Book-Title'].values[0],
                  book_data[book_data['Book-Title'] == book_dataset.to_raw_iid(rec[0])]['Image-URL-L'].values[0])
                 for rec in recommendations]

    return top_books

# Streamlit UI
st.title("Book Recommender System")

# Get unique book titles
book_titles = book_data['Book-Title'].unique()

# Select book
selected_book = st.selectbox("Select a book:", book_titles)

if st.button('Recommend'):
    recommendations = recommend(selected_book)
    st.write("## Recommended Books:")
    for book_title, book_image_url in recommendations:
        col1, col2 = st.columns(2)
        with col1:
            st.image(book_image_url, use_column_width=True)
        with col2:
            st.write(book_title)