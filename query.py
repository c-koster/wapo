"""
YES -- I now know how to USE an existing index rather than making a new one every time.
major key to getting this to work.
"""


from whoosh.qparser import QueryParser
from whoosh.index import open_dir

ix = open_dir("indexdir")

def search_with_terms(text_terms):
    """

    """
    d = []
    with ix.searcher() as searcher:
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

def get_by_id(article_id):
    """
    Most useful as I already have pairwise judgements
    """
    with ix.searcher() as searcher:
        article = searcher.document(path=article_id)
    return article

if __name__ == "__main__":
    print(search_with_terms("hello")[0])
    print(get_by_id("31d8e582-3a3e-11e1-9d6b-29434ee99d6a"))
