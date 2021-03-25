"""
Should be a way to see the relevant articles, and eventually view the best background links'
below. Flask app for UI here we go
"""

from flask import Flask, render_template
import json
import gzip
from query import search_with_terms, get_by_id

app = Flask(__name__)

@app.route("/")
def index():
    # get a list of article id and pass them into the index template
    links = search_with_terms("hello")
    print(links)

    return render_template("index.html",links=links)


@app.route("/article/<string:article_id>")
def article(article_id):
    # 1. query using article_id on the article i'm looking for
    doc = get_by_id(article_id)

    # 2. check if we found the article
    if doc == None: # if we didn't find it render a 404 page
        return render_template("error.html",msg={'code':404,'text':'404 not found'})

    # 3. if we *did* find it, search for some relevant articles using Whoosh
    links = [{"id":"bcdbc240-30d3-11e1-8c61-c365ccf404c5"}, {"id":"74a81038-2da8-11e1-8af5-ec9a452f0164"},
             {"id":"ca152d58-3adc-11e1-9ff8-fab9392b31bf"}, {"id":"93c63d52-3c74-11e1-af18-7ec0de5907e2"}]

    links = search_with_terms(doc['title'])

    # 4. then apply *my model* to the search results and rank them in decreasing order of relevance
    # ( this will be the big assignment )

    # 5. Finally use the links and article to render the page.
    return render_template("article.html",article=doc,links=links)
