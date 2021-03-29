"""
YES -- I now know how to USE an existing index rather than making a new one every time.
major key to getting this to work.
"""


from whoosh.qparser import QueryParser
from whoosh.index import open_dir
from functools import lru_cache # foley caching magic

ix = open_dir("indexdir")
searcher = ix.searcher()
# so as it turns out -- the open and closing of a searcher are quite computationally expensive
# defining a searcher as a global variable here speeds up feature extraction of 42,000 pairs
# from 9 hours to about 10 seconds


def search_with_terms(text_terms):
    """
    Perform a search on the indexed files
    """
    d = []
    parser = QueryParser("content", schema=ix.schema)
    query = parser.parse(text_terms)
    results = searcher.search(query)
    for r in results:
        d.append(r.fields())
        #print (r, r.score)
        # Was this results object created with terms=True?
        if results.has_matched_terms():
            # What terms matched in the results?
            print(results.matched_terms())
    return d

@lru_cache(maxsize=1000) # i am speed
def get_by_id(article_id):
    """
    get an article by its id -- most useful as I already have pairwise judgements
    """

    article = searcher.document(path=article_id)
    return article

if __name__ == "__main__":
    print(search_with_terms("hello")[0])
    print(get_by_id("31d8e582-3a3e-11e1-9d6b-29434ee99d6a"))
