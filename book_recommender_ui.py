import streamlit as st
import pickle
import numpy as np
import joblib

def recommend_book(book):
    index = np.where(pt.index==book)[0][0]
    similar_books = sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1],reverse=True)[1:6]
    
    data = []
    for i in similar_books:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        data.append(item)
    return data

st.header("Book Recommender System")
books = joblib.load('books.joblib')
pt = joblib.load('pt.joblib')
similarity_score = joblib.load('similarity_score.joblib')
final_ratings = joblib.load('final_ratings.joblib')

book_list = pt.index.values


st.sidebar.title("Recommend Books")
selected_book = st.sidebar.selectbox("Type or select a book from the dropdown",book_list)
if st.sidebar.button("Recommend"):
    book_result = recommend_book(selected_book)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(book_result [0][2])
        st.text(book_result[0][0])
        st.text(book_result [0][1])  
    with col2:
        st.image(book_result[1][2])
        st.text(book_result[1][0])
        st.text(book_result[1][1])
    with col3:
        st.image(book_result[2][2])
        st.text(book_result[2][0])
        st.text(book_result[2][1])
    with col4:
        st.image(book_result[3][2])
        st.text(book_result[3][0])
        st.text(book_result[3][1])
    with col5:
        st.image(book_result[4][2])
        st.text(book_result[4][0])
        st.text(book_result[4][1])