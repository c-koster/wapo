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

# PART 2: maybe I get to writing some code?

from query import search_with_terms, get_by_id
from bs4 import BeautifulSoup as bs


def extract_features(id1, id2):
    """
    1. pull the text of both these documents from the index
    """
    a1 = get_by_id(id1)
    a2 = get_by_id(id2)
    # do some text magic

    # same author?
    # size of union of their features (jaccard index [0,1])
    # distance in time
    # categorical stuff
    # TFIDF vectorizor dot product --> similarity

    pass # return a feature vector


def get_labels_list(filenames):
    # build a dictionary to map query numbers to doc_ids
    query_map = {}
    labels = []
    count_positives = 0

    for filename in filenames:
        with open("queries/{}".format(filename),'r') as queries_file:

            # Combine the lines in the list into a string
            content = queries_file.readlines()
            content = "".join(content)

            # load the string into bs4
            bs_content = bs(content, "lxml")
            elems = bs_content.find_all("top")

            for e in elems: # loop over the xml data that we found
                id = e.find("docid").text # get the doc id
                num_map = e.find("num").text[-4:-1] # pluck the query number from this string

                # and add these to the dict so  num_map -> id
                query_map[num_map] = id

    for i in ["18","19","20"]: # for each of the years
        with open("queries/newsir{}-background-linking.qrel".format(i),'r') as labels_file:
            for line in labels_file:
                line_stripped = line.strip().split(' ')

                label = int(line_stripped[3]) > 0 # this is a true/false value
                seed_article = query_map[line_stripped[0]] # get the article id from query number

                labels.append({'similar': label, 'id_0': seed_article, 'id_1': line_stripped[2]})

    # i need a list of labels of the form
    # {'similar': score, 'id_0':'12345','id_1':'12344'}
    print(count_positives)
    return labels

labels = get_labels_list(["newsir18-background-linking-topics.xml","newsir19-entity-ranking-topics.xml","newsir20-topics.xml"])
# the query-map files have different names, so pass in their names here
