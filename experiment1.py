#%%
import gzip, json
from tqdm import tqdm
import typing as T
from dataclasses import dataclass, field
import re
import numpy as np

# get sklearn in here --
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from sklearn.metrics import average_precision_score
from sklearn.base import RegressorMixin

# and the models we're going to try -- regression ends up not being so good
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor



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


def clean(d: T.Dict[str,T.Any]) -> T.Dict[str,T.Any]:
    if 'path' in d.keys():
        d['id'] = d['path']
        del d['path']

    if 'pool-score' in d.keys():
        del d['pool-score']

    return d


def extract_features(left: T.Dict[str,T.Any],right: T.Dict[str,T.Any]) -> T.Dict[str,T.Any]:

    left = clean(left)
    right = clean(right)

    qdoc = WapoArticle(**left)
    doc = WapoArticle(**right)

    q_title = set(tokenize(qdoc.title))
    q_words = tokenize(qdoc.body)
    q_uniq_words = set(q_words)

    doc_title = set(tokenize(doc.title))
    words = tokenize(doc.body)
    uniq_words = set(words)
    avg_word_len = safe_mean([len(w) for w in words])
    features = {
        "time-delta": qdoc.published_date - doc.published_date,
        "title-sim": jaccard(q_title, doc_title),
        "title-body-sim": jaccard(q_title, uniq_words),
        "title-body-sim-rev": jaccard(doc_title, q_uniq_words),
        "avg_word_len": avg_word_len,
        "length": len(words),
        "uniq_words": len(uniq_words),
    }
    return features



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
    model: RegressorMixin

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
        qdoc = query["doc"]

        data = RankingData(qid)
        rank = 1
        for entry in query["pool"]:
            features = extract_features(left=qdoc, right=entry["doc"])
            features["pool-score"] = 1/rank
            truth = entry["truth"]

            data.examples.append(features)
            data.docids.append(entry['doc']['id'])
            data.labels.append(truth)
            rank +=1

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

# combine data:
train = collect("train", train_qs, qids_to_data)

# convert to matrix and scale features:
numberer = train.fit_vectorizer()
fscale = StandardScaler()
X_train = fscale.fit_transform(train.get_matrix(numberer))


def compute_query_APs(model: RegressorMixin, dataset: T.List[str]) -> T.List[float]:
    ap_scores = []
    # eval one query at a time:
    for qid in dataset:
        query = qids_to_data[qid]
        X_qid = fscale.transform(query.get_matrix(numberer))
        qid_scores = model.predict(X_qid)
        # AP uses binary labels:
        labels = query.get_ys() > 0
        if True not in labels:
            # about four queries that have no positive judgments
            continue
        AP = average_precision_score(labels, qid_scores)
        ap_scores.append(AP)
    return ap_scores


def consider_forest() -> ExperimentResult:

    performances: T.List[ExperimentResult] = [] # try a bunch and keep the best one !

    for rnd in tqdm(range(3)): # 3 random restarts
        for crit in ["mse"]:
            for d in range(4,7):
                params = {
                    "criterion": crit,
                    "max_depth": d,
                    "random_state": rnd,
                }
                f = RandomForestRegressor(**params)
                f.fit(X_train, train.get_ys())
                vali_ap = np.mean(compute_query_APs(f,vali_qs))
                result = ExperimentResult(vali_ap, params, f)
                performances.append(result)

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
            f = SGDRegressor(**params)
            f.fit(X_train, train.get_ys())
            vali_ap = np.mean(compute_query_APs(f,vali_qs))
            result = ExperimentResult(vali_ap, params, f)
            performances.append(result)
    return max(performances, key=lambda result: result.vali_ap)

def consider_knn() -> ExperimentResult:

    performances: T.List[ExperimentResult] = []
    for w in ["distance","uniform"]:
        for num in [5,50,500]:
            params = {
                "weights": w,
                "n_neighbors": num
            }
            f = KNeighborsRegressor(**params)
            f.fit(X_train, train.get_ys())
            vali_ap = np.mean(compute_query_APs(f,vali_qs))
            result = ExperimentResult(vali_ap, params, f)
            performances.append(result)
    return max(performances, key=lambda result: result.vali_ap)



# a 'regression' model for each document is usually __NOT__ amazing.
# it's considered the worst way to do it.
result = consider_forest()
keep_model = result.model # the random forest is definitely best
#linear_model = consider_linear().model
#knn = consider_knn().model

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

# then save it for my live implementation:
from joblib import dump
dump(keep_model, 'model.joblib')

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

        m = RandomForestRegressor(**result.params)
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
plt.savefig("learning-curves.png")
plt.show()
