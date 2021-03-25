"""
Updated Foley's collect code to write index files into whoosh's search system.
Also I included here the two functions I'll need to search through these articles.

"""
import gzip, json
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID, DATETIME
from whoosh.qparser import QueryParser

import os.path

schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT,date=DATETIME)
ix = create_in("indexdir", schema)

def search(text_terms):

    return


def get_by_id(article_id):
    with ix.searcher() as searcher:
         query = QueryParser("content", ix.schema).parse("is")
         results = searcher.search(query)
         print(results[0])

         for r in results:
             print (r, r.score)
             # Was this results object created with terms=True?
             if results.has_matched_terms():
                # What terms matched in the results?
                print(results.matched_terms())

         # What terms matched in each hit?
         print ("matched terms")
         for hit in results:
            print(hit.matched_terms())
    return




if __name__ == '__main__':
    if not os.path.exists("indexdir"):
        os.mkdir("indexdir")

    writer = ix.writer()
    writer.add_document(title=u"First document", path=u"/a",content=u"This is the first document we've added!")

    keep = set()
    with open('queries/all.ids', 'r') as fp:
        for line in fp:
            keep.add(line.strip())

    found = 0
    with open('trec.wapo.mini.jsonl', 'w') as out:
        with gzip.open('TREC_Washington_Post_collection.v3.jl.gz', 'rt') as fp:
            for line in fp:
                doc = json.loads(line)
                if doc['id'] in keep:
                    writer.add_document(title=u"hello world",content=u"this is a test")
                    found += 1
                    print(found)
                    # stop rule for now
                    if found > 100:
                        break
    writer.commit()


    get_by_id()
