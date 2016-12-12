"""
ENG* 6500 Final Project
The purpose of this project is to determine the best way to represent
multi-word terms inside a support vector machine for the purpose of
tag prediction
Written BY: AJ Olpin
Written On: December 1st 2016
"""

import json
import math
import numpy as np
import warnings
import sys
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing as pp
from sklearn import svm
from sklearn import metrics as ms
from sklearn.model_selection import train_test_split

class DataProcess(object):
    random_seed = random.randint(1000,7000)
    max_lines = 1000
    max_terms = 25

    def load_data_bow(self, data_file):
        print 'Processing data file %s...' % data_file
        y_data = []
        x_dataTerms = []
        classes = []
        count = 0
        with open(data_file, 'rb') as infile:
            for line in infile:
#                if count == self.max_lines:
#                    break
                jrecord = json.loads(line)
                y_data.append(jrecord['topics'])
                for topic in jrecord['topics']:
                    classes.append(topic)
                x_dataTerms.append(' '.join(jrecord['terms']))
                count += 1
        classes = list(set(classes))

        return x_dataTerms, y_data, classes

    def load_data_norm(self, data_file):
        print 'Processing data file %s...' % data_file
        y_data = []
        x_dataTerms = []
        classes = []
        count = 0
        with open(data_file, 'rb') as infile:
            for line in infile:
#                if count == self.max_lines:
#                    break
                probabilities = []
                jrecord = json.loads(line)
                y_data.append(jrecord['topics'])
                for topic in jrecord['topics']:
                    classes.append(topic)
                i = 0
                for term in jrecord['terms']:
                    for key in term:
                        token = key.replace(' ',"-")
                        probabilities.append(token)
                        i += 1
                joined_term = " ".join(probabilities)
                x_dataTerms.append(joined_term)
                count += 1
        classes = list(set(classes))

        return x_dataTerms, y_data, classes

    def load_data_score(self, data_file):
        print 'Processing data file %s...' % data_file
        y_data = []
        x_termsScores = []
        x_allterms = []
        classes = []
        count = 0
        totalTerms = 0
        with open(data_file, 'rb') as infile:
            for line in infile:
#                if count == self.max_lines:
#                    break
                probabilities = {}
                jrecord = json.loads(line)
                y_data.append(jrecord['topics'])
                for topic in jrecord['topics']:
                    classes.append(topic)
                term_insert_count = 0
                for term in jrecord['terms']:
                    for key,value in term.iteritems():
                        probabilities[key] = value
                        if not key in x_allterms:
                            x_allterms.append(key)
                    term_insert_count += 1
                    if term_insert_count >= self.max_terms:
                        break
                totalTerms += len(probabilities)
                x_termsScores.append(probabilities)
                count += 1
        classes = list(set(classes))

        return x_termsScores, x_allterms, y_data, classes

    def load_data_log(self, data_file):
        print 'Processing data file %s...' % data_file
        y_data = []
        x_dataLists = []
        x_dict = {}
        classes = []
        total_terms = 0;
        linecount = 0
        with open(data_file, 'rb') as infile:
            for line in infile:
#                if linecount == self.max_lines:
#                    break
                term_dict = {}
                term_list = []
                jrecord = json.loads(line)
                y_data.append(jrecord['topics'])
                for topic in jrecord['topics']:
                    classes.append(topic)
                #Add every term to the term_dictionary
                for term in jrecord['terms']:
                    for key,value in term.iteritems():
                        if not term_dict.has_key(key):
                            term_dict[value] = key

                #Add only the top N values
                term_insert_count = 0
                key_list = sorted(list(term_dict.keys()),key=float,reverse=True)
                while term_insert_count < self.max_terms and term_insert_count < len(key_list):
                    term = term_dict[key_list[term_insert_count]]
                    for k in term.split(" "):
                        if x_dict.has_key(k):
                            x_dict[k] += 1
                        else:
                            x_dict[k] = 1
                        total_terms += 1
                    term_list.append(term)
                    term_insert_count += 1
                x_dataLists.append(term_list)
                linecount += 1
        classes = list(set(classes))

        return x_dataLists, x_dict ,y_data, classes, total_terms

    def split_data(self, x_data, y_targets, num_train=0.7):

        x_train, x_val, y_train, y_val = train_test_split(x_data, y_targets, train_size=num_train, random_state=self.random_seed)

        return x_train, x_val, y_train, y_val

class Classifier(object):
    label_binarizer = None
    classifier = None
    flags = {'C':0.99, 'loss':'squared_hinge', 'penalty':'l2', 'tol':0.1, 'iterations':1000, 'multi':'ovr', 'inter':4.0}
    random_seed = random.randint(1000, 7000)

    def create_binarizer(self, categories):
        self.label_binarizer = pp.MultiLabelBinarizer(classes=categories)

    def create_classifier_pipeline(self):
        self.classifier = Pipeline([('vectorizer', CountVectorizer()), ('tfidf', TfidfTransformer()),
                                    ('classifer',
                                     OneVsRestClassifier(svm.LinearSVC(C=self.flags['C'], loss=self.flags['loss'], \
                                                                       penalty=self.flags['penalty'],
                                                                       tol=self.flags['tol'],
                                                                       max_iter=self.flags['iterations'],
                                                                       multi_class=self.flags['multi'],
                                                                       random_state=self.random_seed,
                                                                       intercept_scaling=self.flags['inter'])))])

    def create_classifier_tokenizer(self):
        self.classifier = Pipeline([('vectorizer', CountVectorizer(tokenizer=self.tokenize)), ('tfidf', TfidfTransformer()),
                                    ('classifer',
                                     OneVsRestClassifier(svm.LinearSVC(C=self.flags['C'], loss=self.flags['loss'],
                                                                       penalty=self.flags['penalty'],
                                                                       tol=self.flags['tol'],
                                                                       max_iter=self.flags['iterations'],
                                                                       multi_class=self.flags['multi'],
                                                                       random_state=self.random_seed,
                                                                       intercept_scaling=self.flags['inter'])))])

    def create_classifier_vector(self):
        self.classifier = OneVsRestClassifier(svm.LinearSVC(C=self.flags['C'], loss=self.flags['loss'], \
                                                                       penalty=self.flags['penalty'],
                                                                       tol=self.flags['tol'],
                                                                       max_iter=self.flags['iterations'],
                                                                       multi_class=self.flags['multi'],
                                                                       random_state=self.random_seed,
                                                                       intercept_scaling=self.flags['inter']))

    def tokenize(self, text):
        return text.strip().lower().split(" ")

    def binarizer_categories(self):
        return np.array(self.label_binarizer.classes_)

    def binarizer(self, y_data):
        binary = self.label_binarizer.fit_transform(y_data)
        return binary

    def train_model(self, xtrain, ytrain, xval, yval):
        self.classifier.fit(xtrain, ytrain)
        valpredict = self.classifier.predict(xval)
        accuracy = ms.accuracy_score(valpredict,yval)
        precision = ms.precision_score(valpredict,yval,average='weighted')
        recall = ms.recall_score(valpredict,yval,average="weighted")
        coverage = ms.coverage_error(valpredict,yval)
        loss = ms.label_ranking_loss(valpredict,yval)

        return accuracy, precision, recall, coverage, loss

    def predict(self, x_test):
        predicted = self.classifier.predict(x_test)
        self.label_binarizer.inverse_transform(predicted)

        return predicted

    def stats(self, predicted, y_train, labels):
        print(ms.classification_report(y_train, predicted, target_names=labels))
        precision = ms.precision_score(predicted, y_train, average='weighted')
        recall = ms.recall_score(predicted, y_train, average="weighted")
        print ("Weighted: Precision: {:0.2f} Recall: {:0.2f}".format(precision, recall))
        precision = ms.precision_score(predicted, y_train, average='micro')
        recall = ms.recall_score(predicted, y_train, average="micro")
        print ("Micro: Precision: {:0.2f} Recall: {:0.2f}".format(precision, recall))
        precision = ms.precision_score(predicted, y_train, average='macro')
        recall = ms.recall_score(predicted, y_train, average="macro")
        print ("Macro: Precision: {:0.2f} Recall: {:0.2f}".format(precision, recall))
        accuracy = ms.accuracy_score(predicted, y_train)
        print ("Accruacy: {:0.2f}".format(accuracy))

def log_formula_additive(x_list, x_occurs, total, alpha):
    vector_space = np.zeros([len(x_list), len(x_occurs)])

    key_list = list(x_occurs.keys())

    for item in xrange(len(x_list)):
        #print ("Processing {:}".format(item))
        for occur in xrange(len(key_list)):
            val = 0
            curren_list_term = key_list[occur]
            for collection in x_list[item]:
                terms = collection.split(" ")
                if curren_list_term in terms:
                    term_val = 0
                    for track_pos in xrange(len(terms)):
                        tf = (float(x_occurs[terms[track_pos]]) / float(total))
                        log = (alpha * (1.0/(1.0+math.log(track_pos+1.0))))
                        term_val += (tf * log)
                    val += (term_val)
            vector_space[item][occur] = val

    return vector_space

def log_formula_multiply(x_list, x_occurs, total, alpha):
    vector_space = np.zeros([len(x_list), len(x_occurs)])

    key_list = list(x_occurs.keys())

    for item in xrange(len(x_list)):
        #print ("Processing {:}".format(item))
        for occur in xrange(len(key_list)):
            val = 0
            curren_list_term = key_list[occur]
            for collection in x_list[item]:
                terms = collection.split(" ")
                if curren_list_term in terms:
                    if val == 0:
                        val = 1
                    term_val = 0
                    for track_pos in xrange(len(terms)):
                        tf = (float(x_occurs[terms[track_pos]]) / float(total))
                        log = alpha * (1.0 / (1.0 + math.log(track_pos + 1.0)))
                        term_val += (tf*log)
                    val *= (term_val)
            vector_space[item][occur] = val

    return vector_space

def score_vector_space(x_termsScores ,x_allterms):
    vector_space = np.zeros([len(x_termsScores), len(x_allterms)])

    for allterm_pos in xrange(len(x_allterms)):
        print ("Processing {:}".format(allterm_pos))
        for doc_pos in xrange(len(x_termsScores)):
            term = x_allterms[allterm_pos]
            doc_dict = x_termsScores[doc_pos]
            if doc_dict.has_key(term):
                vector_space[doc_pos][allterm_pos] = doc_dict[term]

    return vector_space

"""
This function trains a LinearSVM using  bagofwords and thene gets its results
@:param bagofwords - filepath to a preprocessed json file containing data to use
"""
def run_bagofwords(bagofwords):
    #Load Data
    x_train, y_train, classes = data_processor.load_data_bow(bagofwords)

    # Create binarizer
    classifier.create_binarizer(classes)

    # Split out training, validation and test
    x_train, x_test, y_train, y_test = data_processor.split_data(x_train, y_train)
    x_val, x_test, y_val, y_test = data_processor.split_data(x_test, y_test, 0.5)

    #Create classifier
    classifier.create_classifier_pipeline()

    # Train
    y_val_bin = classifier.binarizer(y_val)
    y_train_bin = classifier.binarizer(y_train)
    accuracy, precision, recall, coverage, loss = classifier.train_model(x_train, y_train_bin, x_val, y_val_bin)

    print (
    "Accuracy: {:0.2f} Precision: {:0.2f} Recall: {:0.2f} Coverage: {:0.2f} Loss: {:0.4f}".format(accuracy, precision,
                                                                                                  recall, coverage,
                                                                                                  loss))
    # Predict
    predicted = classifier.predict(x_test)

    # Compare Results
    y_test_bin = classifier.binarizer(y_test)
    classifier.stats(predicted, y_test_bin, classifier.binarizer_categories())

"""
This method trains an SVM using normalized probability values previously calculated through text mining methods
The method relies on triming vectors to the smallest length vector loaded
@:param filename - the full file path to a json formatted data file to load from
"""
def run_normalized(filename):
    x_train, y_train, classes = data_processor.load_data_norm(filename)

    # Create binarizer
    classifier.create_binarizer(classes)

    # Split out training, validation and test
    x_train, x_test, y_train, y_test = data_processor.split_data(x_train, y_train)
    x_val, x_test, y_val, y_test = data_processor.split_data(x_test, y_test, 0.5)

    # Create classifier
    classifier.create_classifier_tokenizer()

    # Train
    y_val_bin = classifier.binarizer(y_val)
    y_train_bin = classifier.binarizer(y_train)
    accuracy, precision, recall, coverage, loss = classifier.train_model(np.array(x_train), y_train_bin, np.array(x_val), y_val_bin)

    print (
        "Accuracy: {:0.2f} Precision: {:0.2f} Recall: {:0.2f} Coverage: {:0.2f} Loss: {:0.4f}".format(accuracy,
                                                                                                      precision,
                                                                                                      recall, coverage,
                                                                                                      loss))
    # Predict
    predicted = classifier.predict(np.array(x_test))

    # Compare Results
    y_test_bin = classifier.binarizer(y_test)
    classifier.stats(predicted, y_test_bin, classifier.binarizer_categories())

def run_log(filename, additive, alpha):
    x_list, x_occurs, y_train, classes, total = data_processor.load_data_log(filename)

    # Create binarizer
    classifier.create_binarizer(classes)

    if additive:
        vector_space = log_formula_additive(x_list, x_occurs, total, alpha)
    else:
        vector_space = log_formula_multiply(x_list, x_occurs, total, alpha)

    # Split out training, validation and test
    x_train, x_test, y_train, y_test = data_processor.split_data(vector_space, y_train)
    x_val, x_test, y_val, y_test = data_processor.split_data(x_test, y_test, 0.5)

    # Create classifier
    classifier.create_classifier_vector()

    # Train
    y_val_bin = classifier.binarizer(y_val)
    y_train_bin = classifier.binarizer(y_train)
    accuracy, precision, recall, coverage, loss = classifier.train_model(x_train, y_train_bin, x_val, y_val_bin)

    print (
        "Accuracy: {:0.2f} Precision: {:0.2f} Recall: {:0.2f} Coverage: {:0.2f} Loss: {:0.4f}".format(accuracy,
                                                                                                      precision,
                                                                                                      recall, coverage,
                                                                                                      loss))
    # Predict
    predicted = classifier.predict(np.array(x_test))

    # Compare Results
    y_test_bin = classifier.binarizer(y_test)
    classifier.stats(predicted, y_test_bin, classifier.binarizer_categories())

def run_score(filename):
    x_termsScores, x_allterms, y_train, classes = data_processor.load_data_score(filename)

    # Create binarizer
    classifier.create_binarizer(classes)

    vector_space = score_vector_space(x_termsScores, x_allterms)

    # Split out training, validation and test
    x_train, x_test, y_train, y_test = data_processor.split_data(vector_space, y_train)
    x_val, x_test, y_val, y_test = data_processor.split_data(x_test, y_test, 0.5)

    # Create classifier
    classifier.create_classifier_vector()

    # Train
    y_val_bin = classifier.binarizer(y_val)
    y_train_bin = classifier.binarizer(y_train)
    accuracy, precision, recall, coverage, loss = classifier.train_model(x_train, y_train_bin, x_val, y_val_bin)

    print (
        "Accuracy: {:0.2f} Precision: {:0.2f} Recall: {:0.2f} Coverage: {:0.2f} Loss: {:0.4f}".format(accuracy,
                                                                                                      precision,
                                                                                                      recall, coverage,
                                                                                                      loss))
    # Predict
    predicted = classifier.predict(np.array(x_test))

    # Compare Results
    y_test_bin = classifier.binarizer(y_test)
    classifier.stats(predicted, y_test_bin, classifier.binarizer_categories())

def print_parameters(alpha, terms):
    print ("C:{:0.5f} Loss:{:} Tol:{:0.5f} Iterations:{:} multi:{:} inter:{:0.5f} max_terms:{:0.1f} alpha:{:0.1f}".format(
        classifier.flags['C'], classifier.flags['loss'], classifier.flags['tol'], classifier.flags['iterations'],
        classifier.flags['multi'], classifier.flags['inter'], terms, alpha ))

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    #bow = "/Users/aolpin/Developer/MastersWork/TextMining/Project/data/preprocessed-json.txt"
    #avp = "/Users/aolpin/Developer/MastersWork/TextMining/Project/data/multiword-avp-json.txt"
    #square = "/Users/aolpin/Developer/MastersWork/TextMining/Project/data/multiword-square-json.txt"
    square = "/scratch/aolpin/multiword-square-json.txt"
    avp = "/scratch/aolpin/multiword-avp-json.txt"
    bow = "/scratch/aolpin/preprocessed-json.txt"

    #Create classes
    data_processor = DataProcess()
    classifier = Classifier()

    #Get the execution mode
    mode = int(sys.argv[1])
    #mode = 5
    #Check for hyperparamer load mode

    #hyper-parameter mode
    if mode == 0:
        #Set flags from command line
        classifier.flags['C'] = float(sys.argv[2])
        classifier.flags['loss'] = sys.argv[3]
        classifier.flags['tol'] = float(sys.argv[4])
        classifier.flags['iterations'] = int(sys.argv[5])
        classifier.flags['multi'] = sys.argv[6]
        classifier.flags['inter'] = float(sys.argv[7])
        data_processor.max_terms = int(sys.argv[8])
        print_parameters()
    else:
        if mode == 1:
            print_parameters(0,0)
            print ("<------------------Bag Of Words------------------------->")
            run_bagofwords(bow)
            print ("<------------------TF Square------------------------->")
            run_normalized(square)
            print ("<------------------TF AVP------------------------->")
            run_normalized(avp)
        elif mode == 2:
            alpha = int(sys.argv[2])
            data_processor.max_terms = int(sys.argv[3])
            #alpha = 10
            print_parameters(alpha, data_processor.max_terms)
            print ("<------------------Log Square Additive------------------------->")
            run_log(square, True, alpha)
        elif mode == 3:
            alpha = int(sys.argv[2])
            data_processor.max_terms = int(sys.argv[3])
            #alpha = 100
            print_parameters(alpha, data_processor.max_terms)
            print ("<------------------Log Square Multiply------------------------->")
            run_log(square, False, alpha)
        elif mode == 4:
            alpha = int(sys.argv[2])
            data_processor.max_terms = int(sys.argv[3])
            #alpha = 10
            print_parameters(alpha, data_processor.max_terms)
            print ("<------------------Log AVP Additive------------------------->")
            run_log(avp, True, alpha)
        elif mode == 5:
            alpha = int(sys.argv[2])
            data_processor.max_terms = int(sys.argv[3])
            #alpha = 1000
            print_parameters(alpha, data_processor.max_terms)
            print ("<------------------Log AVP Multiply------------------------->")
            run_log(avp, False, alpha)
        elif mode == 6:
            alpha = int(sys.argv[2])
            data_processor.max_terms = int(sys.argv[3])
            print_parameters(0,data_processor.max_terms)
            print ("<------------------Score Square------------------------->")
            run_score(square)
        elif mode == 7:
            alpha = int(sys.argv[2])
            data_processor.max_terms = int(sys.argv[3])
            print_parameters(0,data_processor.max_terms)
            print ("<------------------Score AVP------------------------->")
            run_score(avp)


#Version 2.6

