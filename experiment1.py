#%%
import gzip, json
from tqdm import tqdm
import typing as T
from dataclasses import dataclass, field
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import average_precision_score

#%%

WORD_REGEX = re.compile(r"\w+")


def tokenize(input: str) -> T.List[str]:
    return WORD_REGEX.split(input.lower())


def safe_mean(input: T.List[float]) -> float:
    if len(input) == 0:
        return 0.0
    return sum(input) / len(input)


def jaccard(lhs: T.Set[str], rhs: T.Set[str]) -> float:
    isect_size = sum(1 for x in lhs if x in rhs)
    union_size = len(lhs.union(rhs))
    return isect_size / union_size


@dataclass
class WapoArticle:
    id: str
    title: str
    body: str
    published_date: int
    kicker: T.Optional[str] = None
    url: T.Optional[str] = None
    author: T.Optional[str] = None
    kind: T.Optional[str] = None


@dataclass
class RankingData:
    what: str
    examples: T.List[T.Dict[str, T.Any]] = field(default_factory=list)
    labels: T.List[int] = field(default_factory=list)
    docids: T.List[str] = field(default_factory=list)

    def append(self, other: "RankingData") -> None:
        self.examples.extend(other.examples)
        self.labels.extend(other.labels)
        self.docids.extend(other.docids)

    def fit_vectorizer(self) -> DictVectorizer:
        numberer = DictVectorizer(sort=True, sparse=False)
        numberer.fit(self.examples)
        return numberer

    def get_matrix(self, numberer: DictVectorizer) -> np.ndarray:
        return numberer.transform(self.examples)

    def get_ys(self) -> np.ndarray:
        return np.array(self.labels)


#%%
qids_to_data: T.Dict[str, RankingData] = {}

with gzip.open("pool.jsonl.gz") as fp:
    for line in tqdm(fp, total=160):
        query = json.loads(line)
        qid = query["qid"]
        qdoc = WapoArticle(**query["doc"])

        q_title = set(tokenize(qdoc.title))
        q_words = tokenize(qdoc.body)
        q_uniq_words = set(q_words)

        data = RankingData(qid)

        for entry in query["pool"]:
            doc = WapoArticle(**entry["doc"])
            doc_title = set(tokenize(doc.title))
            truth = entry["truth"]
            words = tokenize(doc.body)
            uniq_words = set(words)
            avg_word_len = safe_mean([len(w) for w in words])
            features = {
                "pool-score": entry["pool-score"],
                # I have a blog-post about why this time-delta is really helpful
                # also, that's what the pool-score is:
                # https://jjfoley.me/2019/07/24/trec-news-bm25.html
                "time-delta": qdoc.published_date - doc.published_date,
                "title-sim": jaccard(q_title, doc_title),
                "title-body-sim": jaccard(q_title, uniq_words),
                "title-body-sim-rev": jaccard(doc_title, q_uniq_words),
                "avg_word_len": avg_word_len,
                "length": len(words),
                "uniq_words": len(uniq_words),
            }

            data.examples.append(features)
            data.docids.append(doc.id)
            data.labels.append(truth)

        qids_to_data[qid] = data

#%%
queries = sorted(qids_to_data.keys())

RANDOM_SEED = 1234567

tv_qs, test_qs = train_test_split(queries, test_size=40 / 160, random_state=RANDOM_SEED)
train_qs, vali_qs = train_test_split(
    tv_qs, test_size=40 / 120, random_state=RANDOM_SEED
)

print("TRAIN: {}, VALI: {}, TEST: {}".format(len(train_qs), len(vali_qs), len(test_qs)))

#%%
def collect(what: str, qs: T.List[str], ref: T.Dict[str, RankingData]) -> RankingData:
    out = RankingData(what)
    for q in qs:
        out.append(ref[q])
    return out


#%%
# combine data:
train = collect("train", train_qs, qids_to_data)

# convert to matrix and scale features:
numberer = train.fit_vectorizer()
fscale = StandardScaler()
X_train = fscale.fit_transform(train.get_matrix(numberer))

# a 'regression' model for each document is usually __NOT__ amazing.
# it's considered the worst way to do it.
m = RandomForestRegressor(max_depth=5, random_state=RANDOM_SEED)
m.fit(X_train, train.get_ys())

# What features are working? (random forests / decision trees are great at this)
print(
    "Feature Importances:",
    sorted(
        zip(numberer.feature_names_, m.feature_importances_),
        key=lambda tup: tup[1],
        reverse=True,
    ),
)



def compute_query_APs(dataset: T.List[str]) -> T.List[float]:
    ap_scores = []
    # eval one query at a time:
    for qid in dataset:
        query = qids_to_data[qid]
        X_qid = fscale.transform(query.get_matrix(numberer))
        qid_scores = m.predict(X_qid)
        # AP uses binary labels:
        labels = query.get_ys() > 0
        if True not in labels:
            # about four queries that have no positive judgments
            continue
        AP = average_precision_score(labels, qid_scores)
        ap_scores.append(AP)
    return ap_scores


print("mAP-train: {:.3}".format(np.mean(compute_query_APs(train_qs))))
print("mAP-vali: {:.3}".format(np.mean(compute_query_APs(vali_qs))))
print("mAP-test: {:.3}".format(np.mean(compute_query_APs(test_qs))))
# %%
