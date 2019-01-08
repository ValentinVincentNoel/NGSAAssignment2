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

nltk.download('punkt')
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()
     
def readCSV(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f)
        return list(reader)

def openLink(filename):
    t_set  = readCSV(filename)
    return [element[0].split(" ") for element in t_set]

training_set = openLink("training_set.txt")
testing_set = openLink("testing_set.txt")

def getInfoNodes():
    node_csv  = readCSV("node_information.csv")
    info_node = {}
    for element in node_csv:
        info_node[element[0]] = {"corpus": element[5], "title": element[2], "author_list": element[3], "time": element[1]}
    return info_node

info_node = getInfoNodes()

# compute TFIDF vector of each paper
vectorizer = TfidfVectorizer(stop_words="english")
features_TFIDF = vectorizer.fit_transform([element["corpus"] for element in info_node.values()])

def randomlySelect(t_set, ratio):
    to_keep = random.sample(range(len(t_set)), k=int(round(len(t_set)*ratio)))
    t_set_reduced = [t_set[i] for i in to_keep]
    labels = np.array([int(element[2]) if len(element) > 2 else None for element in t_set_reduced])
    return t_set_reduced, labels

# we will use three basic features:
# - number of overlapping words in title
# - temporal distance between the papers
#-  number of common authors

def computeFeatures(t_set, info_node, ratio):
    counter = 0
    features = []
    t_set, labels =  randomlySelect(t_set, ratio)
    for i in tqdm(range(len(t_set))):
        source_id = t_set[i][0]
        target_id = t_set[i][1]
        
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

        features.append([overlap_title, temp_diff, comm_auth])

    features = np.array(features)
    # scale
    features = preprocessing.scale(features)

    return features, labels

ratio_training = 0.05
ration_valid = 0.05
m = hashlib.sha256()
m.update((inspect.getsource(computeFeatures)+"_"+str(ratio_training)+"_"+str(ration_valid)).encode("utf-8"))
hash_feature = m.hexdigest()
if os.path.exists(hash_feature):#make sure we have not already computed the features
    def loadJson(name):
        f = open(os.path.join(hash_feature, name+".json"), "r")
        l = list(json.loads(f.read()))
        f.close()
        return l
    training_features = loadJson("train")
    labels = loadJson("train_label")
    validation_features = loadJson("valid")
    labels_validation = loadJson("valid_label")
    testing_features = loadJson("test")
else:
    print("Compute Training features")
    training_features, labels = computeFeatures(training_set, info_node, ratio_training)
    print("Compute Validation features")
    validation_features, labels_validation = computeFeatures(training_set, info_node, ration_valid)
    print("Compute Testing features")
    testing_features, _ = computeFeatures(testing_set, info_node, 1)
    os.mkdir(hash_feature)
    def saveJson(l, name):
        f = open(os.path.join(hash_feature, name+".json"), "w")
        f.write(json.dumps(l.tolist()))
        f.close()
    saveJson(training_features, "train")
    saveJson(labels, "train_label")
    saveJson(validation_features, "valid")
    saveJson(labels_validation, "valid_label")
    saveJson(testing_features, "test")

training_features = np.array(training_features)
validation_features = np.array(validation_features)
testing_features = np.array(testing_features)
labels = np.array(labels)
labels_validation = np.array(labels_validation)


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