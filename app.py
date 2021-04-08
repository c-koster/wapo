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
regressor = load('model.joblib')
# how do I fetch this "pool score" thing?

app = Flask(__name__)


@app.route("/")
def index():
    # get a list of article id and pass them into the index template
    base_query = "health wellness smoothie vegan vegan"
    links,pools = search_with_terms("this gets ignored",base_query)
    return render_template("index.html",links=links)


@app.route("/article/<string:article_id>")
def article(article_id):
    # 1. query using article_id on the article i'm looking for
    qdoc = get_by_id(article_id)
    # 2. check if we found the article
    if qdoc == None: # if we didn't find it render a 404 page
        return render_template("error.html",msg={'code':404,'text':'404 not found'})

    # 3. if we *did* find it, search for some relevant articles using Whoosh
    links, pools = search_with_terms(qdoc['title'],qdoc['body'])
    # question: does links give me access to the
    print(type(links))

    # 4. then apply *my model* to the search results and rank them in decreasing order of relevance
    # ( this will be the big assignment )
    features_X = []
    for doc, score in zip(links, pools):
        print(doc[0])
        f = extract_features(qdoc,doc[0])
        f.update({'pool-score':score})
        features_X.append(f)

    X = numberer.fit_transform(features_X)
    y_pred = regressor.predict(X)
    print(y_pred)

    # 5. Finally use the links and article to render the page.
    return render_template("article.html",article=doc[0],links=[l[0] for l in links])
