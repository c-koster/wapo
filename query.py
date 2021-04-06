"""
YES -- I now know how to use an existing index rather than making a new one every time.
major key to getting this to work.
"""
from whoosh.qparser import QueryParser
from whoosh.index import open_dir
from functools import lru_cache # foley caching magic


from dataclasses import dataclass
from typing import List

from collections import Counter

ix = open_dir("indexdir")
searcher = ix.searcher()
# so as it turns out -- the open and closing of a searcher are quite computationally expensive
# defining a searcher as a global variable here speeds up feature extraction of 42,000 pairs
# from 9 hours to about 10 seconds


# helpers and constants for the search_with_terms function

#from experiment1 import extract_features
stopwords = []
with open('STOPWORDS.txt', 'r') as wordfile:
    lines = wordfile.readlines()
    [stopwords.append(a.strip()) for a in lines]

stopwords = set(stopwords)
print(stopwords)
NUM_TERMS = 50

from sklearn.feature_extraction.text import CountVectorizer
# "CountVectorizer" here and not TfidfVectorizer
word_features = CountVectorizer(strip_accents="unicode",lowercase=True,ngram_range=(1, 1),)


# How do we take take a whole paragraph and turn it into words?
text_to_words = word_features.build_analyzer()

def tokenize(input_text: str) -> List[str]:
    # text_to_words is a function (str) -> List[str]
    return text_to_words(input_text)

@dataclass # marginally better than a tuple
class WeightedTerm:
    weight: float
    token: str


def normalize(wts: List[WeightedTerm]) -> List[WeightedTerm]:
    total = sum(wt.weight for wt in wts)
    return [WeightedTerm(wt.weight / total, wt.token) for wt in wts]


def search_with_terms(title, body):
    """
    Perform a search on the indexed files. In the future i should tune this so
    it accepts a full document, adds up its important words, and performs a
    weighted query.

    I could also have this search take in a qury as a parameter and then just
    call the get_by_id function for the terms that I need
    """
    # copied code structure from prof. foley:

    # 1. tokenize for all the terms -- does it help at all to use the title?
    body_terms = tokenize(body)

    # (need me foor normalization)
    length = len(body_terms)

    # 2. count em up with this dictionary-like counter object
    body_freqs = Counter(body_terms)
    weighted_terms = []

    # 3. create normalised, weighted terms
    for tok, count in body_freqs.items():
        if tok in stopwords:
            continue # don't include this token if it's a stopword

        weighted_terms.append(WeightedTerm(count / length, tok))
    important_first = sorted(weighted_terms, key=lambda wt: wt.weight, reverse=True)

    # 4. keep the top NUM_TERMS (preset to 50) words
    most_important = normalize(important_first[:NUM_TERMS])
    # most_important is a list of type WeightedTerm

    # 5. run the search using (most_important) terms
    wei_query = None

    d = []
    # TODO: I am stuck on what to do here
    parser = QueryParser("content", schema=ix.schema)
    query = parser.parse(title)
    results = searcher.search(query)
    for r in results:
        d.append(r.fields())


    return d

@lru_cache(maxsize=1000) # i am speed
def get_by_id(article_id):
    """
    Retrieve an article by its id.
    """
    article = searcher.document(path=article_id)
    return article

if __name__ == "__main__":
    print(search_with_terms("hello","hello")[0])
    print(get_by_id("31d8e582-3a3e-11e1-9d6b-29434ee99d6a"))
