import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv('books.csv')

books_df = load_data()

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books_df['description'].fillna(''))

# Compute similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to recommend books
def recommend_books(title, cosine_sim=cosine_sim):
    idx = books_df[books_df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    book_indices = [i[0] for i in sim_scores]
    return books_df['title'].iloc[book_indices]

# Streamlit UI
st.title('Book Recommender System')

# Select book
selected_book = st.selectbox('Select a book:', books_df['title'].values)

if st.button('Recommend'):
    recommendations = recommend_books(selected_book)
    st.write('## Recommended Books:')
    for book in recommendations:
        st.write('- ' + book)
