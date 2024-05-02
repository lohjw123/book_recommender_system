import streamlit as st
import pandas as pd
import numpy as np
from surprise import Dataset, Reader
import joblib

# Load the trained model
model = joblib.load('book_recommender_model.joblib')

# Load the books data
books = pd.read_csv("Books.csv")

# Load the ratings data
ratings = pd.read_csv("Ratings.csv")
users = pd.read_csv("Users.csv")

# Preprocess the data
complete_df = ratings.merge(books, on='ISBN')
complete_df.drop(columns=["ISBN", "Image-URL-S", "Image-URL-M"], axis=1, inplace=True)
complete_df = complete_df.merge(users.drop("Age", axis=1), on="User-ID")
complete_df['Location'] = complete_df['Location'].str.split(',').str[-1].str.strip()

# Define the function to get recommendations
def recommend(book_name):
    reader = Reader(rating_scale=(0, 10))
    data = Dataset.load_from_df(complete_df[['User-ID', 'Book-Title', 'Book-Rating']], reader)
    book_dataset = data.build_full_trainset()

    book_id = book_dataset.to_inner_iid(book_name)
    recommendations = model.get_top_n(book_id, n=5)

    data = []

    for recommendation in recommendations:
        book_title = book_dataset.books[recommendation.iid]
        temp_df = books[books['Book-Title'] == book_title]
        item = []
        item.extend(list(temp_df['Book-Title'].values))
        item.extend(list(temp_df['Book-Author'].values))
        item.extend(list(temp_df['Image-URL-M'].values))
        data.append(item)

    return data

# Streamlit UI
st.title("Books Recommendation System")

book_names = books['Book-Title'].unique()
selected_book = st.selectbox("Type or select a book", book_names)

if st.button("Show Recommendation"):
    if selected_book:
        recommendations = recommend(selected_book)
        cols = st.columns(len(recommendations))
        for i, book in enumerate(recommendations):
            with cols[i]:
                st.write(f"**Title:** {book[0]}")
                st.write(f"**Author:** {book[1]}")
                st.image(book[2], use_column_width=True)
    else:
        st.warning("Please select a book.")