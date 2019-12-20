import numpy as np
import pandas as pd

column_names = ['user_id', 'item_id', 'rating', 'timestamp']
#  get Data from dataset
df = pd.read_csv('u_data_test', sep='\t', names=column_names)
# Top 5
# print(df.head())

# get Movie ( ID, Movie title ...)
movie_titles = pd.read_csv("Movie_Id_Titles")
# print(movie_titles.head())

# merge 
df = pd.merge(df,movie_titles,on='item_id')
# print(df.head())

#example of sort highest rating movie vs highest number of rating
# print(df.groupby('title')['rating'].mean().sort_values(ascending=True).head())
# print(df.groupby('title')['rating'].count().sort_values(ascending=False).head())

#create dataframe film with average rating
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
#create col num of ratings and add in the rating df
ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
# print(ratings.head())

# movie matrix 
moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
# print(moviemat.head())

# top movie have most number of rating 
# print(ratings.sort_values("num of ratings", ascending = False ).head(10))

# pick all user rating about that movie
starwars_user_ratings = moviemat['Star Wars (1977)']
# liarliar_user_ratings = moviemat['Liar Liar (1997)']
print(starwars_user_ratings.head())

# use corrwith() method to get correlations between two pandas series
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
# similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)
print(similar_to_starwars.head())

corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
#drop missing value
corr_starwars.dropna(inplace=True)
print(corr_starwars.sort_values('Correlation',ascending=False).head(10))

#join number of rating into the df
corr_starwars = corr_starwars.join(ratings['num of ratings'])
print(corr_starwars.head())

#sort only with which movie have more than 100 rate
print(corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head())