import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import streamlit as st
import random

# Load the dataset
url = 'https://raw.githubusercontent.com/devensinghbhagtani/Bollywood-Movie-Dataset/main/IMDB-Movie-Dataset%20(2023-1951).csv'
df = pd.read_csv(url)

# Preprocess the data
df = df[['Name', 'Year', 'Genre', 'Overview', 'Director', 'Cast', 'IMDB Rating']]
df = df.dropna()

# Convert categorical data to numerical
df['Genre'] = df['Genre'].astype('category').cat.codes
df['Director'] = df['Director'].astype('category').cat.codes
df['Cast'] = df['Cast'].astype('category').cat.codes

# Define the target variable
df['Hit'] = df['IMDB Rating'].apply(lambda x: 1 if x >= 7 else 0)
df = df.drop(columns=['IMDB Rating'])

# Split the data into training and testing sets
X = df.drop(columns=['Hit'])
y = df['Hit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Load the tokenizer, retriever, and model for RAG
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="custom")
model_rag = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq")

# Function to retrieve relevant information using RAG
def retrieve_info(query):
    inputs = tokenizer(query, return_tensors="pt")
    retrieved_docs = retriever(inputs.input_ids, return_tensors="pt")
    outputs = model_rag.generate(input_ids=inputs.input_ids, context_input_ids=retrieved_docs.context_input_ids)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return response

# Function to generate random data
def get_box_office_data(movie_name):
    return f"Estimated box office earnings: ${random.randint(10, 300)} million."

def get_critic_reviews(movie_name):
    return f"Critic reviews: {random.choice(['Positive', 'Mixed', 'Negative'])}."

def get_audience_reviews(movie_name):
    return f"Audience reviews: {random.choice(['Positive', 'Mixed', 'Negative'])}."

def get_social_media_trends(movie_name):
    return f"Social media mentions: {random.randint(1000, 100000)}."

def get_search_trends(movie_name):
    return f"Google search trends: {random.randint(1000, 100000)} searches."

def get_advanced_booking_data(movie_name):
    return f"Advanced bookings: {random.randint(10000, 500000)} tickets."

# Streamlit app
st.title("Bollywood Movie Hit or Flop Predictor")

movie_name = st.text_input("Enter Movie Name:")
year = st.number_input("Enter Release Year:", min_value=1951, max_value=2023, step=1)
genre = st.text_input("Enter Genre:")
director = st.text_input("Enter Director:")
cast = st.text_input("Enter Cast:")

if st.button("Predict"):
    genre_code = df['Genre'].astype('category').cat.categories.get_loc(genre)
    director_code = df['Director'].astype('category').cat.categories.get_loc(director)
    cast_code = df['Cast'].astype('category').cat.categories.get_loc(cast)

    input_data = pd.DataFrame([[movie_name, year, genre_code, director_code, cast_code]],
                              columns=['Name', 'Year', 'Genre', 'Director', 'Cast'])

    query = f"Tell me about the movie {movie_name} releasing in {year} directed by {director} and starring {cast}."
    additional_info = retrieve_info(query)

    box_office_data = get_box_office_data(movie_name)
    critic_reviews = get_critic_reviews(movie_name)
    audience_reviews = get_audience_reviews(movie_name)
    social_media_trends = get_social_media_trends(movie_name)
    search_trends = get_search_trends(movie_name)
    advanced_booking_data = get_advanced_booking_data(movie_name)

    prediction = model.predict(input_data.drop(columns=['Name']))
    result = "Hit" if prediction[0] == 1 else "Flop"

    st.write(f"The movie '{movie_name}' is predicted to be a {result}.")
    st.write(f"Additional Information: {additional_info}")
    st.write(f"Box Office Data: {box_office_data}")
    st.write(f"Critic Reviews: {critic_reviews}")
    st.write(f"Audience Reviews: {audience_reviews}")
    st.write(f"Social Media Trends: {social_media_trends}")
    st.write(f"Google Search Trends: {search_trends}")
    st.write(f"Advanced Booking Data: {advanced_booking_data}")



st.title("Bollywood Movie Hit or Flop Predictor")

movie_name = st.text_input("Enter Movie Name:")
year = st.number_input("Enter Release Year:", min_value=1951, max_value=2023, step=1)
genre = st.text_input("Enter Genre:")
director = st.text_input("Enter Director:")
cast = st.text_input("Enter Cast:")

if st.button("Predict"):
    genre_code = df['Genre'].astype('category').cat.categories.get_loc(genre)
    director_code = df['Director'].astype('category').cat.categories.get_loc(director)
    cast_code = df['Cast'].astype('category').cat.categories.get_loc(cast)

    input_data = pd.DataFrame([[movie_name, year, genre_code, director_code, cast_code]],
                              columns=['Name', 'Year', 'Genre', 'Director', 'Cast'])

    query = f"Tell me about the movie {movie_name} releasing in {year} directed by {director} and starring {cast}."
    additional_info = retrieve_info(query)
