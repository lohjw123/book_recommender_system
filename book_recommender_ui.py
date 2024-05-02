import streamlit as st
import pandas as pd
import joblib
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the model
model = joblib.load('book_recommender_model.joblib')

# Load the datasets
books = pd.read_csv("Books.csv")

# Function to get recommendations
def recommend_books(book_name):
    index = books[books['Book-Title'] == book_name].index[0]
    similarity_scores = model.qi.dot(model.qi[index])
    similar_books_indices = similarity_scores.argsort()[-6:-1][::-1]
    similar_books = books.iloc[similar_books_indices][['Book-Title', 'Book-Author', 'Image-URL-M']]
    similarity_scores = similarity_scores[similar_books_indices]
    return similar_books, similarity_scores

# Streamlit UI
st.title('Book Recommender System')

book_name = st.text_input('Enter a book title:', 'A Painted House')
if st.button('Recommend'):
    recommendations, similarity_scores = recommend_books(book_name)
    st.write('Top 5 Recommended Books:')
    st.text('')
    for i in range(len(recommendations)):
        st.image(recommendations.iloc[i]['Image-URL-M'], caption=recommendations.iloc[i]['Book-Title'])
        st.write('Author:', recommendations.iloc[i]['Book-Author'])
        st.write('Similarity Score:', similarity_scores[i])
        st.text('')