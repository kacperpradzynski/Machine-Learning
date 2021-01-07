import pandas as pd
from scipy import spatial
import operator

def binary(values_list, unique_list):
    binaryList = []
    
    for unique in unique_list:
        if unique in values_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    
    return binaryList

def prepare_genres(movies):
    movies['genres'] = movies['genres'].str.strip('[]').str.replace(' ','').str.replace("'",'')
    movies['genres'] = movies['genres'].str.split(',')

    genreList = []
    for index, row in movies.iterrows():
        genres = row["genres"]
        for genre in genres:
            if genre not in genreList:
                genreList.append(genre)
    movies['genres_bin'] = movies['genres'].apply(lambda x: binary(x, genreList))

def prepare_cast(movies):
    movies['cast'] = movies['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'')
    movies['cast'] = movies['cast'].str.split(',')

    for i,j in zip(movies['cast'],movies.index):
        list2 = []
        list2 = i[:4]
        movies.loc[j,'cast'] = str(list2)
    movies['cast'] = movies['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'')
    movies['cast'] = movies['cast'].str.split(',')

    castList = []
    for index, row in movies.iterrows():
        cast = row["cast"]
        for i in cast:
            if i not in castList:
                castList.append(i)
    movies['cast_bin'] = movies['cast'].apply(lambda x: binary(x, castList))

def prepare_director(movies):
    directorList=[]
    for i in movies['director']:
        if i not in directorList:
            directorList.append(i)
    movies['director_bin'] = movies['director'].apply(lambda x: binary(x, directorList))

def prepare_keywords(movies):
    movies['keywords'] = movies['keywords'].str.strip('[]').str.replace(' ','').str.replace("'",'').str.replace('"','')
    movies['keywords'] = movies['keywords'].str.split(',')

    words_list = []
    for index, row in movies.iterrows():
        keywords = row["keywords"]
        for keyword in keywords:
            if keyword not in words_list:
                words_list.append(keyword)
    movies['words_bin'] = movies['keywords'].apply(lambda x: binary(x, words_list))

def Similarity(movieId1, movieId2):
    a = movies.iloc[movieId1 - 1]
    b = movies.iloc[movieId2 - 1]
    
    genresA = a['genres_bin']
    genresB = b['genres_bin']
    genreDistance = spatial.distance.cosine(genresA, genresB)
    
    castA = a['cast_bin']
    castB = b['cast_bin']
    castDistance = spatial.distance.cosine(castA, castB)
    
    directA = a['director_bin']
    directB = b['director_bin']
    directDistance = spatial.distance.cosine(directA, directB)
    
    wordsA = a['words_bin']
    wordsB = b['words_bin']
    wordsDistance = spatial.distance.cosine(wordsA, wordsB)
    return genreDistance + directDistance + castDistance + wordsDistance

def getNeighbors(baseMovie, rated_movies, K):
        distances = []
    
        for index, movie in rated_movies.iterrows():
            dist = Similarity(int(baseMovie['movieid']), int(movie['movieid']))
            distances.append((int(movie['movieid']), dist, movie['rate']))
    
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
    
        for x in range(K):
            neighbors.append(distances[x])
        return neighbors

movies = pd.read_csv('movies.csv')
prepare_genres(movies)
prepare_cast(movies)
prepare_director(movies)
prepare_keywords(movies)

train = pd.read_csv('./train.csv', sep=';', names=["id", "userid", "movieid", "rate"])
task = pd.read_csv('./task.csv', sep=';', names=["id", "userid", "movieid", "rate"])

for index, row in task.iterrows():
    print(str(index) + " / " + str(len(task.index)), end ='\r')
    neighbors = pd.DataFrame(getNeighbors(row, train.loc[train['userid'] == row["userid"]], 10), columns=['movieid', 'dist', 'rate'])
    task.loc[index, 'rate'] = str(int(round(neighbors['rate'].mean())))

task.to_csv('submission.csv', sep=';', index=False, header=False)