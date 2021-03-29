# PART 1: questions for Foley
# 1. I'd like to process my data in a way that is most conducive to producing features for a model. I'm using collect.py
#   to feed into a search index and also to create json long format files like you said. But it's broken up by paragraphs
#   with html conntent inside.

# 2. All I've really done is make a UI... but my vision for this is that I index the 42,000 articles that you sent me the ids
#   for, search one by ID when you click on it, load you into a fake washington post page with that article's content. Then this
#   would start another search using whoosh, and I'd tune my results with the 'model'. Does this make sense? So far I can't get any
#   results to return so there's a roadblock there

# 3. the 'model' -- here's what my thinking is on this so far. You've suggested a model for answering the question:
#   'are these two articles similar?' which would take in two articles and return a score for how relevent they are to eachother.
#   I guess this means I need to make labels to tell my model which articles are similar and which are not. My labels would then
#   look like {'similar':score, 'id_0':'12345','id_1':'12344'} and then I'd need to make a function extract_features which takes in two ids
#   and returns a vector with features describing what is similar about them.

# 4. this would then be used on the second whoosh search using article1 as a seed article, the search results as article_2, and would
#   extract common feaures between them to make a prediction.


# takeaways so far:
#   search is hard
#   I don't actually have natural labels and need to get started ASAP

# takeaways from foley meeting:
#   there is a competiton in may
#   I DO actually have natural labels, and as i suspected the labels look like this:
#   {'similar': score, 'id_0':'12345','id_1':'12344'}
#   I need a feature creation function for a PAIR of articles, and will learn more about feature engineering soon
#

# PART 2: maybe I get to writing some code

from query import search_with_terms, get_by_id
from bs4 import BeautifulSoup as bs
import datetime as dt

from distance import jaccard

assert jaccard("ABC","ABD") == 0.5


from sklearn.feature_extraction.text import CountVectorizer


# get a list of word_features from larger text block
word_features = CountVectorizer(
    strip_accents="unicode",
    lowercase=True,
    ngram_range=(1, 1), # try 1, 2 or 1, 3 for larger word couplings
)

text_to_words = word_features.build_analyzer()



def extract_features(id1, id2):
    """
    1. pull the text of both these documents from the index

    2. ensure they are not None (meaning the referenced id is not in the db)

    3. then extract some features -- see foley's lectures on feature engineering
    but the idea is to find things which might predict article similarity:
        - same author
        - jaccard index
        - difference in time
        - tdfidf vector dot product
    """
    a1 = get_by_id(id1)
    a2 = get_by_id(id2)
    assert a1
    assert a2

    # same author?
    same_author = (a1['author'] == a2['author'])

    # distance in time
    time1 = dt.datetime.strptime(a1['date'], '%Y-%m-%d %H:%M:%S')
    time2 = dt.datetime.strptime(a2['date'], '%Y-%m-%d %H:%M:%S')
    diff_time = (time2 - time1).days

    # size of union of their features (jaccard index [0,1]):
    # formally |A i B| / |A u B|
    a1_title = text_to_words(a1['title'])
    a2_title = text_to_words(a2['title'])

    jaccard_title = 1 - jaccard(a1_title, a2_title)

    a1_text = text_to_words(a1['content'])
    a2_text = text_to_words(a2['content'])

    jaccard_full = 1 - jaccard(a1_text, a2_text)


    # categorical stuff
    # TFIDF vectorizor dot product --> similarity

    d = {
    'jaccard_full': jaccard_full, 'diff_time': diff_time,
    'same_author': same_author, 'jaccard_title': jaccard_title
    }
    return d # return a dict with all the relevant features we extracted

def get_labels_list(query_map):
    # build a dictionary to map query numbers to doc_ids
    labels = []


    for i in ["18","19","20"]: # for each of the years
        with open("queries/newsir{}-background-linking.qrel".format(i),'r') as labels_file:
            for line in labels_file:
                line_stripped = line.strip().split(" ")

                label = int(line_stripped[3]) > 0 # this is a true/false value
                query_num = line_stripped[0]
                seed_article = query_map[query_num] # get the article id from query number

                labels.append({'similar': label, 'id_0': seed_article, 'id_1': line_stripped[2],'query_num':query_num})

    # i need a list of labels of the form
    # {'similar': score, 'id_0':'12345','id_1':'12344'}
    #print(count_positives)
    return labels

def build_query_map(filenames):
    query_map = {}

    for filename in filenames:
        with open("queries/{}".format(filename),'r') as queries_file:

            # Combine the lines in the list into a string
            content = queries_file.readlines()
            content = "".join(content)

            # load the string into bs4
            bs_content = bs(content, "lxml")
            elems = bs_content.find_all("top")

            for e in elems: # loop over the xml data that we found
                id = e.find("docid").text.strip(" ") # get the doc id
                num_map = e.find("num").text[-4:-1] # pluck the query number from this string

                # and add these to the dict so  num_map -> id
                query_map[num_map] = id

    return query_map


# PART 3: train a classifier. Here's how it's going to look:


# 1. first I need train_cv_test splits, but it's kind of cheating if I valiidate or test
#   on the same queries i'll be trainning with. so I need to randomly sample by
#   queries and get labels/features from there. Or as foley suggested, it might

# 2. get my list in terms of X and y's

# 3. try a few models


# the query-map files have different names, so pass in their names here
query_map = build_query_map(["newsir18-background-linking-topics.xml","newsir19-entity-ranking-topics.xml","newsir20-topics.xml"])
#print(query_map.keys())



# split query_map.keys() into a train cv and test
from sklearn.model_selection import train_test_split

RANDOM_SEED = 123

queries_all = list(query_map.keys())

# separate train/validate from test set
queries_tv, queries_test = train_test_split(
    queries_all, train_size=0.75, shuffle=True, random_state=RANDOM_SEED
)

# and thenn separate train and validate queries
queries_train, queries_vali = train_test_split(
    queries_tv, train_size=0.66, shuffle=True, random_state=RANDOM_SEED
)


exit(0)
labels = get_labels_list(query_map)
features = []
# the extract_features function usually fails the assert statements because
# I don't? have? all? the? articles? indexed?
for label_i in labels:
    features.append(extract_features(label_i['id_0'], label_i['id_1']))
    print(features)
