"""
I re-worked Foley's collect code to write index files into whoosh's search system.
"""
import gzip, json
import os.path
import datetime as dt

from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID, NUMERIC

# parse_doc needs this
import typing as T
from dataclasses import dataclass, field


from tqdm import tqdm


@dataclass # i kept this aroundd to easily deal with null or missing values
class WapoArticle:
    id: str
    title: str
    body: str
    published_date: int
    kicker: T.Optional[str] = None
    url: T.Optional[str] = None
    author: T.Optional[str] = None
    kind: T.Optional[str] = None


def create_index():
    """


    """
    if not os.path.exists("indexdir"):
        os.mkdir("indexdir")

    schema = Schema(title = TEXT(stored=True), path = ID(stored=True),
                    author= TEXT(stored=True), body = TEXT(stored=True),
                    published_date = NUMERIC(stored=True), kind = TEXT(stored=True),
                    kicker= TEXT(stored=True))

    ix = create_in("indexdir", schema)
    written_ids = set()

    with ix.writer(procs=4,limitmb=4096,multisegment=True) as writer: # enter into the writer and also the articles file
        with gzip.open("pool.jsonl.gz") as fp:
            for line in tqdm(fp, total=160):

                query = json.loads(line)
                qid = query["qid"]
                qdoc = WapoArticle(**query["doc"])
                if qdoc.id not in written_ids:
                    written_ids.add(qdoc.id)
                    writer.add_document(title=qdoc.title, body=qdoc.body,
                                        path=qdoc.id, author=qdoc.author,
                                        published_date=qdoc.published_date//1000,
                                        kicker=qdoc.kicker,kind=qdoc.kind
                    )

                for entry in query["pool"]:
                    doc = WapoArticle(**entry["doc"])
                    if doc.id not in written_ids:
                        written_ids.add(doc.id)
                        writer.add_document(title=doc.title, body=doc.body,
                                            path=doc.id, author=doc.author,
                                            published_date=doc.published_date//1000,
                                            kicker=doc.kicker,kind=doc.kind
                        )
        print("end of fp loop.")
    print("closed the writer.")

if __name__ == '__main__':

    create_index()

    print("\n\nTrying a few queries:")
    from query import search_with_terms, get_by_id
    print(search_with_terms("hello","body"))
    print(get_by_id("31d8e582-3a3e-11e1-9d6b-29434ee99d6a"))
