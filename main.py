from bs4 import BeautifulSoup
import requests
import praw
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

def clean(text):
    nltk.download('stopwords')
    stopword=set(stopwords.words('english'))
    stemmer = nltk.SnowballStemmer("english")
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

def get_posts(user):
  reddit = praw.Reddit(
    client_id="cznO3iIkmeYrb-erL9QhXQ",
    client_secret="3_CNe1qZikS8WdQQTsKt67pgo9rJGA",
    password="Major07Bella",
    user_agent="script by u/Emotional_Total_8846",
    user="Emotional_Total_8846",
    check_for_async=False
  )
  test_user = reddit.redditor(str(user))
  sub = test_user.submissions.new(limit=None)
  self_texts = []
  for link in sub:
    self_texts.append(link.selftext)
  return clean(self_texts)

def response(user):
  response = requests.get("https://github.com/ColinJ69/miniature-happiness/raw/main/Book%20(1)%20(3).xlsx")
  data = pd.read_excel(response.content,usecols = [0,1])
  
  x = np.array(data["Text"])
  y = np.array(data["Label"])
  
  cv = CountVectorizer()
  X = cv.fit_transform(x)
  xtrain, xtest, ytrain, ytest = train_test_split(X, y,
                                                  test_size=0.2,
                                                  random_state=42)
  
  
  data["Label"] = data["Label"].map({0: "No sad", 1: "Sad"})
  
  
  model = BernoulliNB()
  fit = model.fit(xtrain, ytrain)
  
  
  
  lit = []
  for i in get_posts(user):
    data = cv.transform([i]).toarray()
    output = fit.predict(data)
    lit.append(output.item())
  return lit


if __name__ == '__main__':
  x = input()
  print(response(x))
    
