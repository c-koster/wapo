import gzip, json

import re
import numpy as np
import random
from joblib import load, dump
# fancy python stuff
from tqdm import tqdm
import typing as T
from dataclasses import dataclass, field

# get sklearn in here --
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from sklearn.metrics import average_precision_score, ndcg_score
from sklearn.base import ClassifierMixin

# and the models we're going to try -- regression ended up not being so good
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from scipy.spatial import distance


# define a dataclass to manage word embeddings
@dataclass
class NamedVectors:
    name_to_row: T.Dict[str, int] = field(default_factory=dict)
    vectors: np.ndarray = np.zeros(1)

    def get(self, name: str) -> T.Optional[np.ndarray]:
        row = self.name_to_row.get(name, -1)
        if row == -1:
            return None
        return self.vectors[row,:]


# need to put it after the NamedVectors object definition or
# else we don't know what that object means
wapo_vectors_raw = load('wapo.distilbert.joblib') # load this enormous 5gb file into memory.


# load in my word embeddings
body = wapo_vectors_raw['body']
vecs_body = NamedVectors(body.name_to_row, body.vectors)
title = wapo_vectors_raw['title']
vecs_title = NamedVectors(title.name_to_row, title.vectors)
ID = "96ab542e-6a07-11e6-ba32-5a4bf5aad4fa"
TOP_ID = "TVJOOPLOWFHMJMDLVTVPPVQCM4"

WORD_REGEX = re.compile(r"\w+")

RANDOM_SEED = 1234567
random.seed(RANDOM_SEED)

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


def clean(d: T.Dict[str,T.Any]) -> T.Dict[str,T.Any]:
    if 'path' in d.keys():
        d['id'] = d['path']
        del d['path']
    if 'pool-score' in d.keys():
        del d['pool-score']
    return d


def extract_features(left: T.Dict[str,T.Any], right: T.Dict[str,T.Any]) -> T.Dict[str,T.Any]:

    # this is so my live imp and my experiments can use the same function.
    left = clean(left)
    right = clean(right)

    # make everyone into a class so I can use object notation
    qdoc = WapoArticle(**left)
    doc = WapoArticle(**right)

    q_title = set(tokenize(qdoc.title))
    q_words = tokenize(qdoc.body)
    q_uniq_words = set(q_words)

    #q_named = get_named_entities(qdoc.body)
    #doc_named = get_named_entities(doc.body)
    qvec_body = body.get(qdoc.id)
    dvec_body = body.get(doc.id)


    #qvec_title = title.get(qdoc.id)
    #dvec_title = title.get(doc.id)

    doc_title = set(tokenize(doc.title))
    words = tokenize(doc.body)
    uniq_words = set(words)
    avg_word_len = safe_mean([len(w) for w in words])
    features = {
        "time-delta": qdoc.published_date - doc.published_date,
        "title-sim": jaccard(q_title, doc_title),
        "body-cos-distance": distance.cosine(qvec_body, dvec_body),
        #"title-cos-distance": distance.cosine(qvec_title, dvec_title),
        #"named-entity-sim":jaccard(doc_named,q_named),
        "title-body-sim": jaccard(q_title, uniq_words),
        "title-body-sim-rev": jaccard(doc_title, q_uniq_words),
        "body-body-sim": jaccard(uniq_words, q_uniq_words),
        "avg_word_len": avg_word_len,
        "length": len(words),
        "uniq_words": len(uniq_words),
        "author-eq": qdoc.author == doc.author,
        #"random": random.random(),
    }
    return features


# object definitions -- dataclasses, very cool!!
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
class ExperimentResult:
    vali_ap: float
    params: T.Dict[str, T.Any]
    model: ClassifierMixin

@dataclass
class RankingData:
    what: str
    examples: T.List[T.Dict[str, T.Any]] = field(default_factory=list)
    labels: T.List[int] = field(default_factory=list)
    docids: T.List[str] = field(default_factory=list)
    isClassifier: bool = False

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
        a = np.array(self.labels)
        if self.isClassifier:
            a = a > 0
        return a


if __name__ == "__main__":

    qids_to_data: T.Dict[str, RankingData] = {}

    with gzip.open("pool.jsonl.gz") as fp:
        for line in tqdm(fp, total=160):
            query = json.loads(line)
            qid = query["qid"]
            qdoc = query["doc"]

            data = RankingData(qid,isClassifier=True)
            rank = 1
            for entry in query["pool"]:
                features = extract_features(left=qdoc, right=entry["doc"])
                features["pool-rank"] = 1/rank
                truth = entry["truth"]

                data.examples.append(features)
                data.docids.append(entry['doc']['id'])
                data.labels.append(truth)
                rank +=1

            qids_to_data[qid] = data

    queries = sorted(qids_to_data.keys())


    tv_qs, test_qs = train_test_split(queries, test_size=40 / 160, random_state=RANDOM_SEED)
    train_qs, vali_qs = train_test_split(
        tv_qs, test_size=40 / 120, random_state=RANDOM_SEED
    )

    print("TRAIN: {}, VALI: {}, TEST: {}".format(len(train_qs), len(vali_qs), len(test_qs)))


    def collect(what: str, qs: T.List[str], ref: T.Dict[str, RankingData],excludeFeature: str="") -> RankingData:

        out = RankingData(what,isClassifier=True)
        for q in qs:
            out.append(ref[q])
        if excludeFeature != "":
            x = 0 # no-op

        return out

    # combine data:
    train = collect("train", train_qs, qids_to_data)

    # convert to matrix and scale features:
    numberer = train.fit_vectorizer()
    fscale = StandardScaler()
    X_train = fscale.fit_transform(train.get_matrix(numberer))


    def compute_query_APs(model: ClassifierMixin, dataset: T.List[str]) -> T.List[float]:
        ap_scores = []
        # eval one query at a time:
        for qid in dataset:
            query = qids_to_data[qid]
            X_qid = fscale.transform(query.get_matrix(numberer))
            # this outputs two values --we want the
            qid_scores = model.predict_proba(X_qid)[:,1]
            # AP uses binary labels:
            labels = query.get_ys() > 0
            if True not in labels:
                # about four queries that have no positive judgments
                continue
            AP = average_precision_score(labels, qid_scores)
            ap_scores.append(AP)
        return ap_scores


    def compute_query_ndcgs(model: ClassifierMixin, dataset: T.List[str], depth=10) -> T.List[float]:
        ap_scores = []
        # weights = np.array([0, 1, 2, 3, 4]) ** 2
        # eval one query at a time:
        for qid in dataset:
            query = qids_to_data[qid]
            X_qid = fscale.transform(query.get_matrix(numberer))
            if hasattr(model, "decision_function"):
                qid_scores = model.decision_function(X_qid)
            elif hasattr(model, "predict_proba"):
                # qid_scores = (weights * m.predict_proba(X_qid)).sum(axis=1)
                qid_scores = model.predict_proba(X_qid)[:, 1]
            else:  # must be regression
                qid_scores = model.predict(X_qid)
            # AP uses binary labels:
            labels = query.get_ys()
            if labels.sum() == 0.0:
                # about four queries that have no positive judgments
                continue
            ndcg10 = ndcg_score(
                y_true=np.array([labels]), y_score=np.array([qid_scores]), k=depth
            )
            ap_scores.append(ndcg10)
        return ap_scores



    def consider_forest() -> ExperimentResult:

        performances: T.List[ExperimentResult] = [] # try a bunch and keep the best one !

        for rnd in tqdm(range(3)): # 3 random restarts
            for crit in ["gini","entropy"]:
                for d in [2, 4, 7, 10, None]:
                    for leafsize in [2]:
                        params = {
                            "criterion": crit,
                            "max_depth": d,
                            "random_state": rnd,
                            "min_samples_leaf": leafsize
                        }
                        f = RandomForestClassifier(**params)
                        #f = ExtraTreesClassifier(**params)
                        f.fit(X_train, train.get_ys())
                        vali_ap = np.mean(compute_query_APs(f,vali_qs))
                        result = ExperimentResult(vali_ap, params, f)
                        performances.append(result)

        """
        p = {'criterion': 'gini', 'max_depth': 7, 'random_state': 1, 'min_samples_leaf': 2}
        f = RandomForestClassifier(**p)
        f.fit(X_train, train.get_ys())
        vali_ap = np.mean(compute_query_APs(f,vali_qs))
        result = ExperimentResult(vali_ap, p, f)
        performances.append(result)
        """

        # return the model with the best performanece
        return max(performances, key=lambda result: result.vali_ap)


    def consider_linear() -> ExperimentResult:
        # same deal here

        performances: T.List[ExperimentResult] = []

        for rnd in range(3): # 3 random restarts
            for p in ["l1","l2"]:
                params = {
                    "random_state": rnd,
                    "penalty": p
                }
                f = SGDClassifier(**params)
                f.fit(X_train, train.get_ys())
                vali_ap = np.mean(compute_query_APs(f,vali_qs))
                result = ExperimentResult(vali_ap, params, f)
                performances.append(result)
        return max(performances, key=lambda result: result.vali_ap)



    # a 'regression' model for each document is usually __NOT__ amazing.
    # it's considered the worst way to do it.
    # - here i've switched to a binary classifier instead, and provide my
    # pointwise score with predict_proba
    result = consider_forest()
    keep_model = result.model # the random forest is definitely best
    #linear_model = consider_linear().model
    print(result.params)

    # What features are working? (random forests / decision trees are great at this)
    print(
        "Feature Importances:",
        sorted(
            zip(numberer.feature_names_, keep_model.feature_importances_),
            key=lambda tup: tup[1],
            reverse=True,
        ),
    )
    print("mAP-train: {:.3}".format(np.mean(compute_query_APs(keep_model, train_qs))))
    print("mAP-vali: {:.3}".format(np.mean(compute_query_APs(keep_model, vali_qs))))
    print("mAP-test: {:.3}".format(np.mean(compute_query_APs(keep_model,test_qs))))

    print("ndcgs: {:.3}".format(np.mean(compute_query_ndcgs(keep_model, test_qs))))


    # then save it for my live implementation:
    dump(keep_model, 'model.joblib')



    # many of my features are repetetive, so here I'm going to try a feature removal analysis
    # by adding a method to the collect function
    b = collect("train", train_qs, qids_to_data)

    # convert to matrix and scale features:
    numberer = b.fit_vectorizer()
    for name in numberer.feature_names_:
        print(name)


    exit(0)

    # ok as foley suggested, let's try a learning curve anaysis. using the collect function
    num_train = list(range(5, len(train_qs), 5))
    num_train.append(len(train_qs))
    n_trials = 10
    scores = {}

    aps_mean = []
    aps_std = []

    for n_qs in tqdm(num_train):
        label = "queries: {}".format(n_qs)
        scores[label] = []

        for _ in range(n_trials): # do n_trials times
            # subsample train_qs.. do this  n_qs times
            sample_qs = resample(
                train_qs, n_samples=n_qs, replace=False
            )

            train_i = collect("train", sample_qs, qids_to_data)
            # convert to matrix and scale features:
            numberer = train_i.fit_vectorizer()
            fscale = StandardScaler()
            X_train = fscale.fit_transform(train_i.get_matrix(numberer))

            #m = ExtraTreesClassifier(**result.params)
            m = RandomForestClassifier(**result.params)
            m.fit(X_train, train_i.get_ys()) # then fit it

            scores[label].append(np.mean(compute_query_APs(m, vali_qs)))

        aps_mean.append(np.mean(scores[label]))
        aps_std.append(np.std(scores[label]))


    # First, try a line plot, with shaded variance regions:
    import matplotlib.pyplot as plt

    means = np.array(aps_mean)
    std = np.array(aps_std)
    plt.plot(num_train, aps_mean, "o-")
    plt.fill_between(num_train, means - std, means + std, alpha=0.2)
    plt.xlabel("Number of Training Queries")
    plt.ylabel("Mean AP")
    plt.xlim([5, len(train_qs)])
    plt.title("Learning Curve")
    plt.savefig("learning-curves2.png")
    plt.show()

    # if you use random forest classifier and try predict proba -- you get
    #  good/similar results results (e.g. multiclass classifier or regressor)

    # TODO:
    # extract features stuff:
    # - something something transformer distil-bert vector
    # - named entities

    # what makes something a good 'background' article to describe what's going on
    # here... "readability score"
