import streamlit as st
import pandas as pd
import joblib
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the model
model = joblib.load('book_recommender_model.joblib')

# Load the datasets
complete_df = pd.read_csv("Complete.csv")
books = pd.read_csv("Books.csv")

# Function to get recommendations
def recommend_books(book_name):
    index = books[books['Book-Title'] == book_name].index[0]
    similarity_scores = model.qi.dot(model.qi[index])
    similar_books_indices = similarity_scores.argsort()[1:6][::-1]
    similar_books = books.iloc[similar_books_indices][['Book-Title', 'Book-Author', 'Image-URL-M']]
    return similar_books

# Streamlit UI
st.title('Book Recommender System')

book_name = st.text_input('Enter a book title:', 'A Painted House')
if st.button('Recommend'):
    recommendations = recommend_books(book_name)
    st.write('Top 5 Recommended Books:')
    st.text('')
    col_count = 0
    for index, row in recommendations.iterrows():
        st.image(row['Image-URL-M'], caption=row['Book-Title'])
        col_count += 1
        if col_count % 5 == 0:
            st.text('')