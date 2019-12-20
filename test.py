import pandas as pd 
import numpy as np

movie_title = pd.read_csv("Movie_Id_Titles")
print(movie_title.head())
movie_title = movie_title.as_matrix()
listOfMovieTitle = movie_title[:, 1]
listOfSuggestion = [1388, 1493, 1486, 1494, 1618, 599, 1622, 113, 1233, 854]
for i in listOfSuggestion:
    print(listOfMovieTitle[i])
# for n in range(recommended_items):
#             # row indices of rating done by user n
#             # since indices need to be integers, we need to convert
#             ids = np.where(users == n)[0].astype(np.int32)