import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader

# Load the model
model = load('book_recommender_model.joblib')

# Load book data
df_books = pd.read_csv('books.csv')

# Function to recommend books
def recommend(title, model=model):
    # Create a reader
    reader = Reader(rating_scale=(1, 5))

    # Load dataset from DataFrame
    data = Dataset.load_from_df(df_books[['User-ID', 'ISBN', 'Book-Rating']], reader)

    # Build a training set
    trainset = data.build_full_trainset()

    # Make predictions
    isbn = df_books.loc[df_books['Book-Title'] == title, 'ISBN'].iloc[0]
    uid = trainset.to_inner_uid(1)  # Assuming user ID 1 for simplicity
    prediction = model.predict(uid, isbn, verbose=True)
    
    # Retrieve top N recommendations based on the prediction
    # For example, you can use model.test() to get predictions for all items
    # and then sort them based on predicted ratings to get top N recommendations
    # This step depends on your specific implementation and requirements

    return prediction

# Streamlit UI
st.title("Book Recommender System")

# Get unique book titles
book_titles = df_books['Book-Title'].unique()

# Select book
selected_book = st.selectbox("Select a book:", book_titles)

if st.button('Recommend'):
    recommendation = recommend(selected_book)
    st.write("## Recommended Book:")
    st.write("- " + str(recommendation))