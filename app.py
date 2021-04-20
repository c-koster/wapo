"""
Should be a way to see the relevant articles, and eventually view the best background links'
below. Flask app for UI here we go
"""

from flask import Flask, render_template
import json
import gzip
from query import search_with_terms, get_by_id


# fetch everything I need to use for my sklearn model
# these imports are for implementing a live model --
from experiment1 import extract_features
from sklearn.feature_extraction import DictVectorizer
from joblib import load
numberer = DictVectorizer(sort=True, sparse=False)
model = load('model.joblib')
# how do I fetch this "pool score" thing?

app = Flask(__name__)


@app.route("/")
def index():
    # get a list of article id and pass them into the index template
    base_query = "Iran trump trump policy"
    links = search_with_terms("not a news title",base_query)
    return render_template("index.html",links=links)


@app.route("/article/<string:article_id>")
def article(article_id):
    # 1. query using article_id on the article i'm looking for
    qdoc = get_by_id(article_id)
    # 2. check if we found the article
    if qdoc == None: # if we didn't find it render a 404 page
        return render_template("error.html",msg={'code':404,'text':'404 not found'})

    # 3. if we *did* find it, search for some relevant articles using Whoosh
    links = search_with_terms(qdoc['title'],qdoc['body'])
    # question: does links give me access to the

    # 4. then apply *my model* to the search results and rank them in decreasing order of relevance
    # ( this will be the big assignment )

    # 4a. - get the features
    features_X = []
    rank = 1
    for doc in links:
        f = extract_features(qdoc,doc)
        f.update({'pool-score':1/rank}) # use (1/rank) as a feature.
        features_X.append(f)
        rank +=1
    X = numberer.fit_transform(features_X)

    # 4b. apply the model and SORT by its output
    y_pred = model.predict_proba(X)[:,1]
    scores_dict = {}

    for score, link in zip(y_pred, links):
        scores_dict[link['id']] = score
        print(score,link['title'])

    L = sorted(links,key = lambda x: scores_dict[x['id']], reverse=True)

    # 5. Finally use the links and article to render the page.
    return render_template("article.html",article=qdoc,links=L)
