import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import sqrt


def prepare_movies(movies):
  df = movies.copy()
  df['year'] = df.title.str.extract('(\(\d\d\d\d\))',expand=False)
  df['year'] = df.year.str.extract('(\d\d\d\d)',expand=False)
  df['title'] = df.title.str.replace('(\(\d\d\d\d\))', '')
  df['title'] = df['title'].apply(lambda x: x.strip())
  #Every genre is separated by a | so we simply have to call the split function on |
  df['genres'] = df.genres.str.split('|')
  moviesWithGenres_df = df.copy()

  #For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
  for index, row in df.iterrows():
      for genre in row['genres']:
          moviesWithGenres_df.at[index, genre] = 1
  #Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
  moviesWithGenres_df = moviesWithGenres_df.fillna(0)
  moviesWithGenres_df.head()
  return df, moviesWithGenres_df

def preprocess_movies(movies):
  df = movies.copy()
  df['year'] = df.title.str.extract('(\(\d\d\d\d\))',expand=False)
  df['year'] = df.year.str.extract('(\d\d\d\d)',expand=False)
  df['title'] = df.title.str.replace('(\(\d\d\d\d\))', '')
  df['title'] = df['title'].apply(lambda x: x.strip())
  #Every genre is separated by a | so we simply have to call the split function on |
  df = df.drop('genres', 1)
  return df

def prepare_ratings(ratings):
  #Drop removes a specified row or column from a dataframe
  ratings = ratings.drop('timestamp', 1)
  return ratings

def preprocess_movies(movies):
  df = movies.copy()
  df['year'] = df.title.str.extract('(\(\d\d\d\d\))',expand=False)
  df['year'] = df.year.str.extract('(\d\d\d\d)',expand=False)
  df['title'] = df.title.str.replace('(\(\d\d\d\d\))', '')
  df['title'] = df['title'].apply(lambda x: x.strip())
  #Every genre is separated by a | so we simply have to call the split function on |
  df = df.drop('genres', 1)
  return df

def getMovieRecommendation(x, movies, ratings, links, n=20):
  mov, moviesWithGenres = prepare_movies(movies)
  ratings = prepare_ratings(ratings)
  df = ratings.groupby('userId').get_group(x)
  inputId = mov[mov['movieId'].isin(df['movieId'].tolist())]
  inputMovies = pd.merge(inputId, df)
  inputMovies = inputMovies.drop(['genres', 'userId'], 1)
  userMovies = moviesWithGenres[moviesWithGenres['movieId'].isin(inputMovies['movieId'].tolist())]
  #userMovies
  userMovies = userMovies.reset_index(drop=True)
  userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
  #userGenreTable
  userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
  #userProfile
  genreTable = moviesWithGenres.set_index(moviesWithGenres['movieId'])
  genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
  #genreTable.head()
  recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
  #recommendationTable_df.head()
  recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
  #recommendationTable_df.head()
  output = mov.loc[mov['movieId'].isin(recommendationTable_df.head(n).keys())]
  return pd.merge(output,links)

def getFavGenre(x, movies, ratings):
  mov, moviesWithGenres = prepare_movies(movies)
  ratings = prepare_ratings(ratings)
  df = ratings.groupby('userId').get_group(x)
  inputId = mov[mov['movieId'].isin(df['movieId'].tolist())]
  inputMovies = pd.merge(inputId, df)
  inputMovies = inputMovies.drop(['genres', 'year', 'userId'], 1)
  userMovies = moviesWithGenres[moviesWithGenres['movieId'].isin(inputMovies['movieId'].tolist())]
  #userMovies
  userMovies = userMovies.reset_index(drop=True)
  userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
  #userGenreTable
  userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
  pct = [x*100/sum(userProfile) for x in userProfile.to_list()]
  k = pd.Series(pct)
  k.index = userProfile.index
  k = pd.Series(pct)
  k.index = userProfile.index
  k.sort_values(inplace=True,ascending=False)
  return k

def getNeighbours(x, movies, ratings, n=10):
  mov = preprocess_movies(movies)
  ratings = prepare_ratings(ratings)
  df = ratings.groupby('userId').get_group(x)
  inputId = mov[mov['movieId'].isin(df['movieId'].tolist())]
  inputMovies = pd.merge(inputId, df)
  inputMovies = inputMovies.drop(['userId', 'year'], 1)
  userSubset = ratings[ratings['movieId'].isin(inputMovies['movieId'].tolist())]
  userSubsetGroup = userSubset.groupby(['userId'])
  userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)
  userSubsetGroup = userSubsetGroup[0:100]
  pearsonCorrelationDict = {}

  for name, group in userSubsetGroup:
      #Let's start by sorting the input and current user group so the values aren't mixed up later on
      group = group.sort_values(by='movieId')
      inputMovies = inputMovies.sort_values(by='movieId')
      #Get the N for the formula
      nRatings = len(group)
      #Get the review scores for the movies that they both have in common
      temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
      #And then store them in a temporary buffer variable in a list format to facilitate future calculations
      tempRatingList = temp_df['rating'].tolist()
      #Let's also put the current user group reviews in a list format
      tempGroupList = group['rating'].tolist()
      #Now let's calculate the pearson correlation between two users, so called, x and y
      Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
      Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
      Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
      
      #If the denominator is different than zero, then divide, else, 0 correlation.
      if Sxx != 0 and Syy != 0:
          pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
      else:
          pearsonCorrelationDict[name] = 0

  pearsonCorrelationDict.items()

  pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
  pearsonDF.columns = ['similarityIndex']
  pearsonDF['userId'] = pearsonDF.index
  pearsonDF.index = range(len(pearsonDF))
  topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:n]
  return mov, ratings, topUsers

def getSimilarMovies(x, movies, ratings, links):
  mov, rates, topUsers = getNeighbours(x, movies, ratings)  
  topUsersRating=topUsers.merge(rates, left_on='userId', right_on='userId', how='inner')
  topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
  tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
  tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
  recommendation_df = pd.DataFrame()
  recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
  recommendation_df['movieId'] = tempTopUsersRating.index
  recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
  output = mov.loc[mov['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]
  return pd.merge(output,links)

def top_rated(movies,rating, n):
  mov = preprocess_movies(movies)
  ratings = prepare_ratings(rating)
  ratings['rating'][ratings['rating']<4] = np.nan
  ratings.dropna(inplace=True)   
  val = ratings.groupby("movieId").count().reset_index()
  val.sort_values('rating',ascending=False, inplace=True)
  val = val.head(n)
  top = pd.merge(val,mov)
  return top
