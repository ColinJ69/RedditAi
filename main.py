import requests
import praw
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
import math
import json
from flask import Flask, redirect, request, render_template

def get_posts(user):
  credentials = 'client_secrets.json'#store my keys in there

  with open(credentials) as f:
    creds = json.load(f)
  reddit = praw.Reddit(
    client_id=creds['client_id'],
    client_secret=creds['client_secret'],
    user_agent=creds['user_agent'],
    redirect_uri=creds['redirect_uri'],
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
  response = requests.get("personal_dataset.not_public")#small private dataset I made
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
  app = Flask(__name__)
  @app.route('/')
  def main():
    return render_template('main.html', error='')
  @app.route('/result', methods=['GET', 'POST']):
  def result():
  try:
    x = request.files['user']
    response = response(x)
    math_stuff = math.ceil(len(response) - (round(len(response)/4)))
    if sorted(result)[int(math_stuff)] == 0:
      return render_template('okay_profile.html')
    else:
      return render_template('worrying_profile.html')
  except:
    return redirect('main.hmtl', error='Error took place, enter valid user and try again')
    
  
    
    
