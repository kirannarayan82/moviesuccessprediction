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
