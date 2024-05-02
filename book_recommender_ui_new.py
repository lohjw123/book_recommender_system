import streamlit as st
import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split, cross_validate
import joblib
from PIL import Image

# Load data
books = pd.read_csv("Books.csv")
users = pd.read_csv("Users.csv")
ratings = pd.read_csv("Ratings.csv")

# Preprocess data
cutted_length_books = len(books) // 5
books = books[:cutted_length_books]
cutted_length_users = len(users) // 5
users = users[:cutted_length_users]
cutted_length_ratings = len(ratings) // 5
ratings = ratings[:cutted_length_ratings]
ratings_with_book_titles = ratings.merge(books,on='ISBN')
ratings_with_book_titles.drop(columns=["ISBN","Image-URL-S","Image-URL-M"],axis=1,inplace=True)
complete_df = ratings_with_book_titles.merge(users.drop("Age", axis=1), on="User-ID")
complete_df['Location'] = complete_df['Location'].str.split(',').str[-1].str.strip()

# Load trained recommendation model
model = joblib.load('book_recommender_model.joblib')

# Streamlit UI
st.title('Book Recommender System')

# Recommendation based on Collaborative Filtering
st.subheader('Recommendation Based on Collaborative Filtering')
user_id = st.number_input('Enter User ID:', min_value=1, max_value=complete_df['User-ID'].max())
if st.button('Get Recommendations'):
    predictions = model.predict(user_id, complete_df['Book-Title'].unique())
    recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:5]
    col1, col2, col3, col4, col5 = st.columns(5)
    for rec in recommendations:
        book_info = books[books['Book-Title'] == rec.iid].iloc[0]
        with col1:
            image = Image.open(book_info['Image-URL-M'])
            st.image(image, caption=rec.iid, use_column_width=True)
            st.write(rec.iid)