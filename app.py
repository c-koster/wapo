"""
Should be a way to see the relevant articles, and eventually view the best background links'
below. Flask app for UI here we go
"""

from flask import Flask, render_template

import json
import gzip


app = Flask(__name__)


@app.route("/")
def index():
    # get a list of article id and pass them into the index template
    links = []
    num_ids = 0
    with open('queries/all.ids', 'r') as fp:

        for line in fp:
            if num_ids > 50:
                break
            links.append(line.strip())
            num_ids += 1


    return render_template("index.html",links=links)


@app.route("/articles/<str:article_id>")
def article(article_id):
    # 1. query using article_id on the article i'm looking for
    doc = get_article(article_id)


    # 2. check if we found the article
    if doc['id'] == False: # if we didn't find it render a 404 page
        return render_template("error.html")

    # 3. if we *did* find it, search for some relevant articles using Whoosh
    links = ["0000e8a3e952907fac7e965ced9194ba", "00029f4a07559d69995bfc013985924a",
             "00041887f95b15333ae39503610f2de1", "000563252f7569174d0f1a2aff115854"]

    # 4. then apply *my model* to the search results and rank them in decreasing order of relevance
    # ( this will be the big assignment )

    # 5. Finally use the links and article to render the page.
    return render_template("article.html",article_json=article_json,links=links)



# some helper functions ---
def get_article(id):
    with gzip.open('TREC_Washington_Post_collection.v3.jl.gz', 'rt') as fp:
        for line in fp:
            doc = json.loads(line) # json line format.. radical
            if doc['id'] == id:
                return doc

    return {'id':False} # look at this gross typing -- it might be a json XOR a boolean value!!
