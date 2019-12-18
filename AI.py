#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

class CF(object):
    # """docstring for CF
    # input: matrix ['user_id', 'item_id', 'score']
    # k: số item được recommend"""
    def __init__(self, Y_data, k, dist_func = cosine_similarity, uuCF = 1):
        self.uuCF = uuCF # user-user (1) or item-item (0) CF
        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]]
        self.k = k
        self.dist_func = dist_func
        self.Ybar_data = None
        # number of users and items. Remember to add 1 since id starts from 0
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1 
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1
    
    def add(self, new_data):
        # """
        # Update Y_data matrix when new ratings come.
        # For simplicity, suppose that there is no new user or item.
        # """
        self.Y_data = np.concatenate((self.Y_data, new_data), axis = 0)
    
    def normalize_Y(self):
        users = self.Y_data[:, 0] # all users - first col of the Y_data
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))
        for n in range(self.n_users):
            # row indices of rating done by user n
            # since indices need to be integers, we need to convert
            ids = np.where(users == n)[0].astype(np.int32)
            # indices of all ratings associated with user n
            item_ids = self.Y_data[ids, 1] 
            # and the corresponding ratings 
            ratings = self.Y_data[ids, 2]
            # take mean
            m = np.mean(ratings) 
            if np.isnan(m):
                m = 0 # to avoid empty array and nan value
            self.mu[n] = m
            # normalize
            self.Ybar_data[ids, 2] = ratings - self.mu[n]
            ################################################
        # form the rating matrix as a sparse matrix. Sparsity is important 
        # for both memory and computing efficiency. For example, if #user = 1M, 
        # #item = 100k, then shape of the rating matrix would be (100k, 1M), 
        # you may not have enough memory to store this. Then, instead, we store 
        # nonzeros only, and, of course, their locations.
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
            (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
#         print(self.Ybar.tocsr())
        self.Ybar = self.Ybar.tocsr()

    def similarity(self):
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)
        # print("Similarity matrix : ")
        # print(self.S)
    
        
    def refresh(self):
        # """
        # Normalize data and calculate similarity matrix again (after
        # some few ratings added)
        # """
        self.normalize_Y()
        self.similarity() 
        
    def fit(self):
        self.refresh()
        
    
    def __pred(self, u, i, normalized = 1):
        # """ 
        # predict the rating of user u for item i (normalized)
        # if you need the un
        # """
        # Step 1: find all users who rated i
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        # Step 2: 
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        # Step 3: find similarity btw the current user and others 
        # who already rated i
        sim = self.S[u, users_rated_i]
        # Step 4: find the k most similarity users
        a = np.argsort(sim)[-self.k:] 
        # and the corresponding similarity levels
        nearest_s = sim[a]
        # How did each of 'near' users rated item i
        r = self.Ybar[i, users_rated_i[a]]
        if normalized:
            # add a small number, for instance, 1e-8, to avoid dividing by 0
            return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8)

        return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8) + self.mu[u]
    
    def pred(self, u, i, normalized = 1):
        # """ 
        # predict the rating of user u for item i (normalized)
        # if you need the un
        # """
        if self.uuCF: return self.__pred(u, i, normalized)
        return self.__pred(i, u, normalized)
            
    
    def recommend(self, u):
        # """
        # Determine all items should be recommended for user u.
        # The decision is made based on all i such that:
        # self.pred(u, i) > 0. Suppose we are considering items which 
        # have not been rated by u yet. 
        # """
        ids = np.where(self.Y_data[:, 0] == u)[0]
        # items was rated by user u
        items_rated_by_u = self.Y_data[ids, 1].tolist()         
        recommended_items = []
        ratings_items = []
        # check predicting rating for item which user u haven't rated yet, if rating of that item > 0 
        # => add to recommended_items
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                rating = self.__pred(u, i)
                if rating > 0: 
                    ratings_items.append(rating)
                    recommended_items.append(i)

        arr2D = np.array([ratings_items,recommended_items])
        sortedArr = arr2D [ :, arr2D[0].argsort()]
        choosen_item = []
        top_ratings = []
        for i in range(1,11):
            choosen_item.append(int(sortedArr[1][len(ratings_items)-i]))
            top_ratings.append(sortedArr[0][len(ratings_items)-i])
        return choosen_item

    def print_recommendation(self):
        # """
        # print all items which should be recommended for each user 
        # """
        print('Recommendation: ')
        for u in range(self.n_users):
            recommended_items = self.recommend(u)
            if self.uuCF:
#                 temp = [a[i] for i in recommended_items]
                print(' Recommend item(s):', recommended_items , 'for user', u)
            else: 
                print(' Recommend item', u, 'for user(s) : ', recommended_items)



r_cols = ['user_id', 'item_id', 'rating', 'timestam']
# df = pd.read_csv('u.data', sep='\t', names=r_cols)
# df = df.drop(['timestam'], axis=1)

# # Top 5
# print(df.head())

# # get Movie ( ID, Movie title ...)
# movie_titles = pd.read_csv("Movie_Id_Titles")
# # print(movie_titles.head())

# # merge 
# dataframe = pd.merge(df,movie_titles,on='item_id')
# print(dataframe.head())

ratings = pd.read_csv("u.data", sep = '\t', names = r_cols, encoding='latin-1')
ratings = ratings.drop(['timestam'], axis=1)
Y_data = ratings.as_matrix()
print(Y_data)

rs = CF(Y_data, k = 2, uuCF = 1)
rs.fit()

rs.print_recommendation()




# r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
# ratings_base = pd.read_csv('u.data', sep='\t', names=r_cols, encoding='latin-1')
# ratings_test = pd.read_csv('Movie_Id_Titles', sep='\t', names=r_cols, encoding='latin-1')

# rate_train = ratings_base.as_matrix()
# rate_test = ratings_test.as_matrix()

# # indices start from 0
# rate_train[:, :2] -= 1
# rate_test[:, :2] -= 1
# rs = CF(rate_train, k = 30, uuCF = 1)
# rs.fit()

# n_tests = rate_test.shape[0]
# SE = 0 # squared error
# for n in range(n_tests):
#     pred = rs.pred(rate_test[n, 0], rate_test[n, 1], normalized = 0)
#     SE += (pred - rate_test[n, 2])**2 

# RMSE = np.sqrt(SE/n_tests)
# print('User-user CF, RMSE =', RMSE)