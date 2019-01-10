import random
import numpy as np
#import igraph
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn import preprocessing
import nltk
import csv
from tqdm import tqdm
import inspect
import json
import os
import hashlib
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import networkx as nx
from sklearn.metrics import f1_score
from multiprocessing import Pool
from gensim.models import KeyedVectors
     
def readCSV(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f)
        return list(reader)

def openLink(filename):
    t_set  = readCSV(filename)
    return [element[0].split(" ") for element in t_set]


def getInfoNodes():
    node_csv  = readCSV("node_information.csv")
    info_node = {}
    for element in node_csv:
        info_node[element[0]] = {"corpus": element[5], "title": element[2], "author_list": element[3], "time": element[1]}
    return info_node



def randomlySelect(t_set, ratio):
    to_keep = random.sample(range(len(t_set)), k=int(round(len(t_set)*ratio)))
    t_set_reduced = [t_set[i] for i in to_keep]
    labels = np.array([int(element[2]) if len(element) > 2 else None for element in t_set_reduced])
    return t_set_reduced, labels, to_keep



def compute_stats(network, source, target):
    has_edge = False
    if network.has_edge(source, target):
        has_edge = True
        network.remove_edge(source, target)
    if nx.has_path(network, source=source, target=target):
        n_path = 0
        """all_path = nx.all_shortest_paths(network, source=source, target=target)
        for a in all_path:
            n_path +=1"""
        shortest_length =  nx.shortest_path_length(network, source=source, target=target)
    else:
        n_path = 0
        shortest_length = -1
    s_an = nx.all_neighbors(network, source)
    t_an = nx.all_neighbors(network, target)
    s = []
    for i in s_an:
        s.append(i)
    t = []
    for i in t_an:
        t.append(i)
    count = 0
    for elt in s:
        if elt in t:
            count += 1
    n_common = count
    if has_edge:
        network.add_edge(source, target)
    return n_path,shortest_length, n_common

def computeFeatures(t_set):
    counter = 0
    features = []
    for i in tqdm(range(len(t_set))):
        source_id = t_set[i][0]
        target_id = t_set[i][1]

        n_path,shortest_length, n_common = compute_stats(DG, source_id, target_id)
        
        source_info = info_node[source_id]
        target_info = info_node[target_id]

        # convert to lowercase and tokenize
        source_title = source_info["title"].lower().split(" ")
        # remove stopwords
        source_title = [token for token in source_title if token not in stpwds]
        source_title = [stemmer.stem(token) for token in source_title]
        
        target_title = target_info["title"].lower().split(" ")
        target_title = [token for token in target_title if token not in stpwds]
        target_title = [stemmer.stem(token) for token in target_title]
        
        source_auth = source_info["author_list"].split(",")
        target_auth = target_info["author_list"].split(",")
        
        overlap_title = len(set(source_title).intersection(set(target_title)))
        temp_diff = int(source_info["time"]) - int(target_info["time"])
        comm_auth = len(set(source_auth).intersection(set(target_auth)))

        #taille plus court chemin

        features.append([overlap_title, temp_diff, comm_auth, n_path,shortest_length, n_common])

    features = np.array(features)
    # scale
    features = preprocessing.scale(features)

    return features

def saveJson(l, name):
    f = open(os.path.join(hash_feature, name+".json"), "w")
    f.write(json.dumps(l.tolist()))
    f.close()

def loadJson(name):
    f = open(os.path.join(hash_feature, name+".json"), "r")
    l = list(json.loads(f.read()))
    f.close()
    return l

def trainAndTest(classifier, name):
    print(name)
    # train
    print("Training")
    classifier.fit(training_features, labels)

    # validation
    prediction_validation = list(classifier.predict(validation_features))

    def F1Score(gold_label, predict_label):
        tp = 0
        fp = 0
        fn = 0
        for i in range(0, len(gold_label)):
            if gold_label[i] == 1:
                if predict_label[i] == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if predict_label[i] == 1:
                    fp += 1
        p = float(tp)/(float(tp+fp))
        r = float(tp)/(float(tp+fn))
        f1 = 2*p*r/(p+r)
        return p, r, f1

    p, r, f1 = F1Score(labels_validation, prediction_validation)
    print("Validation")
    print("Precision : "+str(p))
    print("Recall : "+str(r))
    print("F1 : "+str(f1))

    # test

    # issue predictions
    predictions_SVM = list(classifier.predict(testing_features))

    # write predictions to .csv file suitable for Kaggle (just make sure to add the column names)
    predictions_SVM = zip(range(len(testing_set)), predictions_SVM)

    with open(os.path.join(hash_feature, name+".csv"),"w",newline='') as pred1:
        csv_out = csv.writer(pred1)
        csv_out.writerow(("id","category"))
        for row in predictions_SVM:
            csv_out.writerow(row)

if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('stopwords')
    stpwds = set(nltk.corpus.stopwords.words("english"))
    stemmer = nltk.stem.PorterStemmer()

    embeddings = KeyedVectors.load_word2vec_format('crawl-300d-2M.vec')

    training_set = openLink("training_set.txt")
    testing_set = openLink("testing_set.txt")

    info_node = getInfoNodes()

    # compute TFIDF vector of each paper
    vectorizer = TfidfVectorizer(stop_words="english")
    features_TFIDF = vectorizer.fit_transform([element["corpus"] for element in info_node.values()])

    # we will use three basic features:
    # - number of overlapping words in title
    # - temporal distance between the papers
    #-  number of common authors

    ratio_training = 0.05
    ration_valid = 0.05

    validation_set, labels_validation, to_keep = randomlySelect(training_set, ration_valid)
    training_set = np.delete(training_set, to_keep, 0)

    DG = nx.DiGraph()
    DG.add_nodes_from(info_node.keys())
    for i in range(len(training_set)):
        if training_set[i][2] == "1":
            DG.add_edge(training_set[i][0],training_set[i][1])

    training_set, labels, to_keep = randomlySelect(training_set, ratio_training)

    m = hashlib.sha256()
    m.update((inspect.getsource(computeFeatures)+"_"+str(ratio_training)+"_"+str(ration_valid)).encode("utf-8"))
    hash_feature = m.hexdigest()
    if os.path.exists(hash_feature):#make sure we have not already computed the features
        training_features = loadJson("train")
        labels = loadJson("train_label")
        validation_features = loadJson("valid")
        labels_validation = loadJson("valid_label")
        testing_features = loadJson("test")
    else:
        print("Compute Training features")
        training_features = computeFeatures(training_set)
        print("Compute Validation features")
        validation_features = computeFeatures(validation_set)
        print("Compute Testing features")
        testing_features = computeFeatures(testing_set)
        #training_features = computeFeatures(DG, training_set, info_node, ratio_training)
    
        #validation_features = computeFeatures(DG, validation_set, info_node, ration_valid)
        
        #testing_features = computeFeatures(DG, testing_set, info_node, 1)
        os.mkdir(hash_feature)
        
        saveJson(training_features, "train")
        saveJson(labels, "train_label")
        saveJson(validation_features, "valid")
        saveJson(labels_validation, "valid_label")
        saveJson(testing_features, "test")


    print(training_features[:10])

    training_features = np.array(training_features)
    validation_features = np.array(validation_features)
    testing_features = np.array(testing_features)
    labels = np.array(labels)
    labels_validation = np.array(labels_validation)


   

    # basic SVM
    classifier = svm.LinearSVC()
    trainAndTest(classifier, "LinearSVC")

    # Random Forest
    classifier = RandomForestClassifier(n_estimators=100, max_depth=3, n_jobs=8)
    trainAndTest(classifier, "RandomForestClassifier")

    # Boosting
    classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    trainAndTest(classifier, "GradientBoostingClassifier")

    # Logistic Regressor
    classifier = LogisticRegression()
    trainAndTest(classifier, "LogisticRegression")

    #LightGBM
    classifier = LGBMClassifier(n_estimators=100, n_jobs=8)
    trainAndTest(classifier, "LightGBM")

    #XGBoost
    classifier = XGBClassifier(n_estimators=100, n_jobs=8)
    trainAndTest(classifier, "XGBoost")