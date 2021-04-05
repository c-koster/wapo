"""
I re-worked Foley's collect code to write index files into whoosh's search system.
"""
import gzip, json
import os.path
import datetime as dt

from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID, DATETIME

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
                    author= TEXT(stored=True), content = TEXT(stored=True),
                    date  = TEXT(stored=True), kind = TEXT(stored=True),
                    kicker= TEXT(stored=True))

    ix = create_in("indexdir", schema)
    written_ids = set()

    writer = ix.writer() # enter into the writer and also the articles file
    with gzip.open("pool.jsonl.gz") as fp:

        for line in tqdm(fp, total=10):

            query = json.loads(line)
            qid = query["qid"]
            qdoc = WapoArticle(**query["doc"])
            if qdoc.id not in written_ids:
                written_ids.add(qdoc.id)
                writer.add_document(title=qdoc.title, content=qdoc.body,
                                    path=qdoc.id, author=qdoc.author,
                                    date=str(dt.datetime.fromtimestamp(qdoc.published_date//1000)),
                                    kicker=qdoc.kicker,kind=qdoc.kind
                )

            for entry in query["pool"]:
                doc = WapoArticle(**entry["doc"])
                if doc.id not in written_ids:
                    written_ids.add(doc.id)
                    writer.add_document(title=doc.title, content=doc.body,
                                        path=doc.id, author=doc.author,
                                        date=str(dt.datetime.fromtimestamp(doc.published_date//1000)),
                                        kicker=doc.kicker,kind=doc.kind
                    )
        print("end of fp loop.")
    writer.commit()
    print("closed the writer.")


if __name__ == '__main__':

    create_index()

    print("\n\nTrying a few queries:")
    from query import search_with_terms, get_by_id
    print(search_with_terms("hello"))
    print(get_by_id("31d8e582-3a3e-11e1-9d6b-29434ee99d6a"))
