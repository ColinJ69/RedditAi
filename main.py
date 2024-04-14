import requests
import praw
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

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
  comment = test_user.comments.new(limit=None)
  self_texts = []
  for link in sub:
    self_texts.append(link.selftext)
  for e in comment:
    self_texts.append(e.body)
  return self_texts

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
  e = get_posts(user)
  for i in e:
    data = cv.transform([i]).toarray()
    output = fit.predict(data)
    lit.append(output.item())
  return lit


if __name__ == '__main__':
  x = input()
  result = response(x)
  j = round(len(result)/4)
  u = (len(result) - j)
  if sorted(result)[int(u)] == 1:
    print("depressed")
  else:
    print("nuh uh")
    
