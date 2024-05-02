import streamlit as st
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
import joblib

# Load the trained model
model = joblib.load('book_recommender_model.joblib')

# Load the books data
books = pd.read_csv("Books.csv")

# Define the function to get recommendations
def recommend(book_name):
    index = np.where(pt.index==book_name)[0][0]
    similar_books = sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1], reverse=True)[1:6]

    data = []

    for i in similar_books:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

        data.append(item)
    return data

# Streamlit UI
st.title("Book Recommendation System")

# Get the user's input for the book name
book_name = st.text_input("Enter the book name:", "")

if book_name:
    recommendations = recommend(book_name)
    st.subheader("Recommended Books:")
    for book in recommendations:
        st.write(f"**Title:** {book[0]}")
        st.write(f"**Author:** {book[1]}")
        st.image(book[2], use_column_width=True)
        st.write("---")