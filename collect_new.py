"""
Foley did some more work on my incomplete data problemss

"""
import gzip, json
from tqdm import tqdm
from dataclasses import dataclass, field

import os.path
import datetime as dt

from query import * # dataclass and search functions


with gzip.open("pool.jsonl.gz") as fp:
    for line in tqdm(fp, total=160):
        query = json.loads(line)
        qid = query["qid"]

        qdoc = query["doc"]
        q_title = qdoc["title"]
        print(qid,q_title)

        for entry in query["pool"]:
            #print(entry.keys())
            print("-" + entry['doc']['title'])
        break
