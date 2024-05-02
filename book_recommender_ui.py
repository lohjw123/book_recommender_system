import streamlit as st
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
import joblib
from sklearn.metrics.pairwise import cosine_similarity

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

# Filter the data to include only active users and popular books
active_user = complete_df.groupby('User-ID')['Book-Rating'].count() > 50
active_user_id = active_user[active_user].index
active_user_ratings = complete_df[complete_df['User-ID'].isin(active_user_id)]
filtered_books = active_user_ratings.groupby('Book-Title').count()['Book-Rating'] > 15
popular_books = filtered_books[filtered_books].index
final_ratings =  active_user_ratings[active_user_ratings['Book-Title'].isin(popular_books)]
final_ratings.drop_duplicates(inplace=True)

# Create the pivot table and calculate the similarity scores
pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)
similarity_score = cosine_similarity(pt)

# Define the function to get recommendations
def recommend(book_name):
    index = np.where(pt.index==book_name)[0][0]
    similar_books = sorted(list(enumerate(similarity_score[index])), key=lambda x:x[1], reverse=True)[1:6]

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

# Add a button to trigger the recommendation process
if st.button("Recommend Books"):
    if book_name:
        recommendations = recommend(book_name)
        st.subheader("Recommended Books:")
        for book in recommendations:
            st.write(f"**Title:** {book[0]}")
            st.write(f"**Author:** {book[1]}")
            st.image(book[2], use_column_width=True)
            st.write("---")
    else:
        st.warning("Please enter a book name.")