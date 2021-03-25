"""
Updated Foley's collect code to write index files into whoosh's search system.
Also I included here the two functions I'll need to search through these articles.

"""
import gzip, json
import os.path
import datetime as dt

from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID, DATETIME

def parse_doc(doc):
    """
    Get a simplified doc dictionary and also run beautifulsoup on the text content
    to get all the HTML yuck out of there.
    """
    # UNIX TIMESTAMP -> datetime object
    # (1325379405000)
    date_pub_timestamp = str(dt.datetime.fromtimestamp(doc['published_date']//1000))
    title = doc['title']
    author = doc['author']
    text = ""
    for c in doc['contents']:
        try:
            if c['subtype'] == 'paragraph':
                text += c['content'] + "\n\n" # NEEDS beautiful soup

        except KeyError:
            continue

    newdict = {'id':doc['id'],'title':title,'author':author,'text':text,'date':date_pub_timestamp}
    return newdict


if __name__ == '__main__':

    if not os.path.exists("indexdir"):
        os.mkdir("indexdir")

    schema = Schema(title = TEXT(stored=True), path = ID(stored=True),
                    author= TEXT(stored=True), content = TEXT(stored=True),
                    date  = TEXT(stored=True))

    ix = create_in("indexdir", schema)

    keep = set()
    with open('queries/all.ids', 'r') as fp:
        for line in fp:
            keep.add(line.strip())

    found = 0
    with ix.writer() as writer:
        with gzip.open('TREC_Washington_Post_collection.v3.jl.gz', 'rt') as fp:

            for line in fp:
                doc = json.loads(line)
                if doc['id'] in keep:
                    parsed_doc = parse_doc(doc)

                    writer.add_document(title=parsed_doc['title'], content=parsed_doc['text'],
                                        path=parsed_doc['id'], author=parsed_doc['author'], date=parsed_doc['date'])

                    found += 1
                    if found % 30 == 0:
                        print(doc['id'])
                    #print(found)
                    if found > 400:
                        break
    # implicit: writer.commit()

    from query import search_with_terms, get_by_id
    print(search_with_terms("hello"))
    print(get_by_id("31d8e582-3a3e-11e1-9d6b-29434ee99d6a"))
