from tmdbv3api import TMDb, Movie
import pandas as pd

tmdb = TMDb()
tmdb.api_key = ''
movie = Movie()

movie_list = pd.read_csv('./movie.csv', sep=';', names=["id", "tmdbid", "title"])
movies = pd.DataFrame(columns=['id', 'genres', 'cast', 'vote_average', 'director', 'keywords'])

map_name= lambda a : a.name
map_original_name= lambda a : a.original_name
for index, row in movie_list.iterrows():
    m = movie.details(movie_id=row['tmdbid'])
    c = movie.credits(movie_id=row['tmdbid'])
    id = row['id']
    genres = list(map(map_name, m.genres))
    cast = list(map(map_original_name, c.cast))
    vote_average = m.vote_average
    director = None
    for crewmate in c.crew:
        if(crewmate.job == "Director"):
            director = crewmate.original_name
    keywords = list(map(map_name, m.keywords.keywords))
    movies.loc[id - 1] = [id, genres, cast, vote_average, director, keywords]
    print("Downloading: " + str(index) + " / " + str(len(movie_list.index)), end ='\r')
movies.to_csv('movies.csv', index=False)