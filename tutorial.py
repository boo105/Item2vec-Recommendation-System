import pandas as pd
import numpy as np

# 데이터 읽기
df_movies = pd.read_csv('./dataset/ml-25m/movies.csv') # 영화 데이터
df_ratings = pd.read_csv('./dataset/ml-25m/ratings.csv') # user의 rating 데이터

# 인코딩
movieId_to_name = pd.Series(df_movies.title.values, index = df_movies.movieId.values)
name_to_movieId = pd.Series(df_movies.movieId.values, index = df_movies.title).to_dict()

# Randomly print 5 records in the dataframe
for df in list((df_movies, df_ratings)):
    rand_idx = np.random.choice(len(df), 5, replace=False)
    # print(df.iloc[rand_idx,:])
    print(df.iloc[rand_idx,:])
    print("Randomly printing 5 of the total "+str(len(df))+" data points")


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
ax = plt.subplot(111)
ax.set_title("Distribution of Movie Ratings", fontsize=16)
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
  
plt.xticks(fontsize=12)  
plt.yticks(fontsize=12)  
  
plt.xlabel("Movie Rating", fontsize=14)  
plt.ylabel("Count", fontsize=14)  
  
plt.hist(df_ratings['rating'], color="#3F5D7D")  

plt.show()


from sklearn.model_selection import train_test_split

df_ratings_train, df_ratings_test= train_test_split(df_ratings,
                                                    stratify=df_ratings['userId'],
                                                    random_state = 15688,
                                                    test_size=0.30)


print("Number of training data: "+str(len(df_ratings_train)))
print("Number of test data: "+str(len(df_ratings_test)))

def rating_splitter(df):
    
    df['liked'] = np.where(df['rating']>=4, 1, 0) # df['rating']이 4이상이면 1 아니면 0
    df['movieId'] = df['movieId'].astype('str')
    gp_user_like = df.groupby(['liked', 'userId']) # liked와 user_id로 그룹화
    return ([gp_user_like.get_group(gp)['movieId'].tolist() for gp in gp_user_like.groups]) 


# 즉, user 정보는 무시하고 movieId만 반환합니다.
pd.options.mode.chained_assignment = None
splitted_movies = rating_splitter(df_ratings_train)


# ***
# ## 모델 생성 및 학습
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
assert gensim.models.word2vec.FAST_VERSION > -1

len(splitted_movies)
new_splitted_movies = splitted_movies[:1000]

# - 영화의 순서는 의미있는 정보가 아니므로 학습데이터를 셔플링합니다.
import random

for movie_list in new_splitted_movies:
    random.shuffle(movie_list)


# 모델을 생성합니다.
# - Skip-gram을 사용한 Word2Vec 모델을 생성합니다.
# - 각 파라미터를 다르게 설정하여 2가지 모델을 만들었습니다.
from gensim.models import Word2Vec
import datetime
start = datetime.datetime.now()

model = Word2Vec(sentences = new_splitted_movies, # We will supply the pre-processed list of moive lists to this parameter
                 epochs = 5, # epoch
                 min_count = 10, # a movie has to appear more than 10 times to be keeped
                 vector_size = 200, # size of the hidden layer
                 workers = 4, # specify the number of threads to be used for training
                 sg = 1, # Defines the training algorithm. We will use skip-gram so 1 is chosen.
                 hs = 0, # Set to 0, as we are applying negative sampling.
                 negative = 5, # If > 0, negative sampling will be used. We will use a value of 5.
                 window = 9999999)

print("Time passed: " + str(datetime.datetime.now()-start))
model.save('item2vec_20210117')
del model

# from gensim.models import Word2Vec
# import datetime
# start = datetime.datetime.now()

# model_w2v_sg = Word2Vec(sentences = splitted_movies,
#                         epochs = 10, # epoch
#                         min_count = 5, # a movie has to appear more than 5 times to be keeped
#                         vector_size = 300, # size of the hidden layer
#                         workers = 4, # specify the number of threads to be used for training
#                         sg = 1,
#                         hs = 0,
#                         negative = 5,
#                         window = 9999999)

# print("Time passed: " + str(datetime.datetime.now()-start))
# model_w2v_sg.save('item2vec_word2vecSg_20210117')
# del model_w2v_sg

# 생성한 모델을 load합니다.
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.models import Word2Vec
model = Word2Vec.load('item2vec_20200908')
word_vectors = model.wv
# del model # uncomment this line will delete the model

import requests
import re
from bs4 import BeautifulSoup

def refine_search(search_term):
    """
    Refine the movie name to be recognized by the recommender
    Args:
        search_term (string): Search Term

    Returns:
        refined_term (string): a name that can be search in the dataset
    """
    target_url = "http://www.imdb.com/find?ref_=nv_sr_fn&q="+"+".join(search_term.split())+"&s=tt"
    
    # 크롤링
    html = requests.get(target_url).content
    parsed_html = BeautifulSoup(html, 'html.parser')
    for tag in parsed_html.find_all('td', class_="result_text"):
        print(tag)
        title = tag.find('a').get_text()
        year = re.findall('</a>(.*)</td>', str(tag))

        return title + ' ' + year[0].strip()
        print(search_result)
        refined_name = ""
        if search_result:
            if search_result[0][0].split()[0]=="The":
                str_frac = " ".join(title.split()[1:])+", "+title.split()[0]
                refined_name = title + ' ' + year[0].strip()
            else:
                refined_name = title + ' ' + year[0].strip()
        return refined_name

def produce_list_of_movieId(list_of_movieName, useRefineSearch=False):
    """
    Turn a list of movie name into a list of movie ids. The movie names has to be exactly the same as they are in the dataset.
    Ambiguous movie names can be supplied if useRefineSearch is set to True
    
    Args:
        list_of_movieName (List): A list of movie names.
        useRefineSearch (boolean): Ambiguous movie names can be supplied if useRefineSearch is set to True

    Returns:
        list_of_movie_id (List of strings): A list of movie ids.
    """
    list_of_movie_id = []
    for movieName in list_of_movieName:
        if useRefineSearch:
            movieName = refine_search(movieName)
            print(movieName)
            print("Refined Name: "+movieName)
        if movieName in name_to_movieId.keys():
            list_of_movie_id.append(str(name_to_movieId[movieName]))
    return list_of_movie_id

def recommender(positive_list=None, negative_list=None, useRefineSearch=False, topn=20):
    recommend_movie_ls = []
    if positive_list:
        positive_list = produce_list_of_movieId(positive_list, useRefineSearch)
    if negative_list:
        negative_list = produce_list_of_movieId(negative_list, useRefineSearch)
    for movieId, prob in model.wv.most_similar_cosmul(positive=positive_list, negative=negative_list, topn=topn):
        recommend_movie_ls.append(movieId)
    return recommend_movie_ls


# recommender
# - user로부터 'UP'이 검색되면 해당 검색어를 크롤링을 통해 dataset의 title값을 받아옵니다.
# - 이후 해당 movie title 리스트를 앞에서 인코딩한 name_to_movieId 사전을 통해 movieId로 바꿔줍니다.
# - 그리고 앞에서 정의한 모델을 통해 top-n 개의 similar한 데이터를 반환합니다.
ls = recommender(positive_list=["UP"], useRefineSearch=True, topn=5)
print('Recommendation Result based on "Up (2009)":')
print(df_movies[df_movies['movieId'].isin(ls)])

# - 모델의 성능을 평가합니다.
def user_liked_movies_builder(model, df, for_prediction=False):
    df['liked'] = np.where(df['rating']>=4, 1, 0)
    df['movieId'] = df['movieId'].astype('str')
    df_liked = df[df['liked']==1]
    if for_prediction:
        df_liked = df[df['movieId'].isin(model.wv.vocab.keys())]
        
    user_liked_movies = df_liked.groupby('userId').agg({'movieId': lambda x: x.tolist()})['movieId'].to_dict()
    
    return user_liked_movies

def scores_at_m (model, user_liked_movies_test, user_liked_movies_training, topn=10):
    sum_liked = 0
    sum_correct = 0
    sum_total = 0
    common_users = set(user_liked_movies_test.keys()).intersection(set(user_liked_movies_training.keys()))

    for userid in common_users:
        current_test_set = set(user_liked_movies_test[userid])
        pred = [pred_result[0] for pred_result in model.wv.most_similar_cosmul(positive = user_liked_movies_training[userid], topn=topn)]
        sum_correct += len(set(pred).intersection(current_test_set))
        sum_liked += len(current_test_set)
    precision_at_m = sum_correct/(topn*len(common_users))
    recall_at_m = sum_correct/sum_liked
    f1 = 2/((1/precision_at_m)+(1/recall_at_m))
    return [precision_at_m, recall_at_m, f1]

pd.options.mode.chained_assignment = None
user_liked_movies_train = user_liked_movies_builder(model, df_ratings_train, for_prediction=True)
user_liked_movies_test = user_liked_movies_builder(model, df_ratings_test)

model = Word2Vec.load('item2vec_20200908')
model_score_sg1 = scores_at_m(model, user_liked_movies_test, user_liked_movies_train)
del model

print("Respectively, the [precision, recall, F-1 score] at 10 for our model are:")
print(model_score_sg1)