#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


ratings = pd.read_csv('https://raw.githubusercontent.com/codeheroku/Introduction-to-Machine-Learning/master/Collaborative%20Filtering/dataset/toy_dataset.csv',index_col = 0)


# In[3]:


ratings = ratings.fillna(0)


# In[4]:


ratings


# In[5]:


def standardize(row):
    new_row = (row - row.mean())/(row.min()-row.max())
    return new_row


# In[6]:


ratings_std = ratings.apply(standardize)


# In[7]:


# we are taking a transpose
item_similarity = cosine_similarity(ratings_std.T)


# In[8]:


item_similarity


# In[9]:


item_similarity_df = pd.DataFrame(item_similarity,index=ratings.columns,columns=ratings.columns)


# In[10]:


item_similarity_df


# In[11]:


item_similarity_df.columns


# In[12]:


## Let's Make Recommendations
def get_similar_movies(movie_name,user_rating):
    similar_score = item_similarity_df[movie_name]*(user_rating-2.5)
    similar_score = similar_score.sort_values(ascending = False)
    return similar_score


# In[13]:


print(get_similar_movies('romantic3',1))


# In[14]:


action_lover = [("action1",5),('romantic2',1),('romantic3',1)]

similar_movies = pd.DataFrame()


# In[15]:


similar_movies


# In[16]:


for movie,rating in action_lover:
    similar_movies = similar_movies.append(get_similar_movies(movie,rating),ignore_index=True)
    


# In[17]:


similar_movies.head()


# In[18]:


similar_movies.sum().sort_values(ascending=False)


# In[19]:


import pandas as pd


# In[20]:


ratings = pd.read_csv('/Users/thenuka/Downloads/dataset 2/ratings.csv')
movies = pd.read_csv('/Users/thenuka/Downloads/dataset 2/movies.csv')
ratings = pd.merge(movies,ratings).drop(['genres','timestamp'],axis=1)
ratings.head()


# In[21]:


user_ratings = ratings.pivot_table(index=['userId'],columns=['title'],values = 'rating')
user_ratings.head()


# In[22]:


# let's Remove Movies which have less than 10 users who rated it. and fill remaining Nan with 0
user_ratings = user_ratings.dropna(thresh=10,axis=1).fillna(0)


# In[23]:


user_ratings


# In[26]:


## Let's Build our similarity Matrix
item_similarity_df = user_ratings.corr(method = 'pearson')


# In[27]:


item_similarity_df.head()


# In[30]:


item_similarity_df.columns


# In[28]:


## Let's Make Recommendations
def get_similar_movies(movie_name,user_rating):
    similar_score = item_similarity_df[movie_name]*(user_rating-2.5)
    similar_score = similar_score.sort_values(ascending = False)
    return similar_score


# In[33]:


romantic_lover = [("(500) Days of Summer (2009)",5),("Alice in Wonderland (2010)",3),("Aliens (1986)",1),("2001: A Space Odyssey (1968)",2)]
similar_movies = pd.DataFrame()
for movie,rating in romantic_lover:
    similar_movies = similar_movies.append(get_similar_movies(movie,rating),ignore_index = True)

similar_movies.head(10)
    


# In[34]:


similar_movies.sum().sort_values(ascending=False).head(20)


# In[ ]:




