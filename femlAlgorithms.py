import os, sys
import time
from abc import ABCMeta, abstractmethod
import math
import re
import itertools

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn import preprocessing
from xgboost import XGBClassifier

from external.datasetcreator import damerau_levenshtein, jaccard, jaro, jaro_winkler, monge_elkan, cosine, strike_a_match, \
    soft_jaccard, sorted_winkler, permuted_winkler, skipgram, davies
# from datasetcreator import detect_alphabet, fields
from staticArguments import perform_stemming, normalize_str, sorted_nicely


class StaticValues:
    algorithms = {
        'damerau_levenshtein': damerau_levenshtein,
        'davies': davies,
        'skipgram': skipgram,
        'permuted_winkler': permuted_winkler,
        'sorted_winkler': sorted_winkler,
        'soft_jaccard': soft_jaccard,
        'strike_a_match': strike_a_match,
        'cosine': cosine,
        'monge_elkan': monge_elkan,
        'jaro_winkler': jaro_winkler,
        'jaro': jaro,
        'jaccard': jaccard,
    }

    methods = [["Damerau-Levenshtein", 0.55],
               ["Jaro", 0.75],
               ["Jaro-Winkler", 0.7],
               ["Jaro-Winkler reversed", 0.75],
               ["Sorted Jaro-Winkler", 0.7],
               ["Permuted Jaro-Winkler", 0.7],
               ["Cosine N-grams", 0.4],
               ["Jaccard N-grams", 0.25],
               ["Dice bigrams", 0.5],
               ["Jaccard skipgrams", 0.45],
               ["Monge-Elkan", 0.7],
               ["Soft-Jaccard", 0.6],
               ["Davis and De Salles", 0.65]]


class FEMLFeatures:
    # TODO to_be_removed = "()/.,:!'"  # check the list of chars
    # Returned vals: #1: str1 is subset of str2, #2 str2 is subset of str1
    @staticmethod
    def contains(strA, strB, sorting=False):
        strA, _ = normalize_str(strA, sorting)
        strB, _ = normalize_str(strB, sorting)
        return set(strA).issubset(set(strB)), set(strB).issubset(set(strA))

    @staticmethod
    def contains_freq_term(self, str, freqTerms=None):
        str, _ = normalize_str(str)
        return True if freqTerms != None and str in freqTerms else False

    def contains_specific_freq_term(self, str):
        pass

    @staticmethod
    def is_matched(str):
        """
        Finds out how balanced an expression is.
        With a string containing only brackets.

        >>> is_matched('[]()()(((([])))')
        False
        >>> is_matched('[](){{{[]}}}')
        True
        """
        opening = tuple('({[')
        closing = tuple(')}]')
        mapping = dict(zip(opening, closing))
        queue = []

        for letter in str:
            if letter in opening:
                queue.append(mapping[letter])
            elif letter in closing:
                if not queue or letter != queue.pop():
                    return False
        return not queue

    def hasEncoding_err(self, str):
        return self.is_matched(str)

    @staticmethod
    def containsAbbr(str):
        abbr = re.search(r"\b[A-Z]([A-Z\.]{1,}|[sr\.]{1,2})\b", str)
        return '-' if abbr is None else abbr.group()

    @staticmethod
    def containsTermsInParenthesis(str):
        tokens = re.split('[{\[(]', str)
        bflag = True if len(tokens) > 1 else False
        return bflag

    def containsDashConnected_words(self, str):
        """
        Hyphenated words are considered to be:
            * a number of word chars
            * followed by any number of: a single hyphen followed by word chars
        """
        is_dashed = re.search(r"\w+(?:-\w+)+", str)
        return False if is_dashed is None else True

    @staticmethod
    def no_of_words(str):
        str, _ = normalize_str(str)
        return len(set(str))

    @staticmethod
    def freq_tokens(tokens, ngram=1):
        if tokens < 1: tokens = 1
        return list(itertools.chain.from_iterable([[tokens[i:i + ngram] for i in range(len(tokens) - (ngram - 1))]]))

    def containsInPos(self, str1, str2):
        fvec_str1 = []
        fvec_str2 = []

        step = math.ceil(len(str1) / 3)
        for idx in xrange(0, len(str1), step):
            if str1[idx:idx + step]:
                sim = damerau_levenshtein(str1[idx:idx + step], str2)
                if sim >= 0.55:
                    fvec_str1.append(1)
                else:
                    fvec_str1.append(0)

        step = math.ceil(len(str2) / 3)
        for idx in xrange(0, len(str2), step):
            if str2[idx:idx + step]:
                sim = damerau_levenshtein(str1, str2[idx:idx + step])
                if sim >= 0.55:
                    fvec_str2.append(1)
                else:
                    fvec_str2.append(0)

        return fvec_str1, fvec_str2

    def fagiSim(self, strA, strB, stop_words):
        # TODO identifyAndExpandAbbr
        # remove punctuations and stopwords, lowercase, sort alphanumerically
        lstrA, _ = normalize_str(strA, sorting=True, stop_words=stop_words)
        lstrB, _ = normalize_str(strB, sorting=True, stop_words=stop_words)
        # TODO extractSpecialTerms
        base, mis = self.compareAndSplit_names(lstrA, lstrB)

    def compareAndSplit_names(self, listA, listB):
        mis = {'A': [], 'B': []}
        base = {'A': [], 'B': []}

        cur = {'A': 0, 'B': 0}
        while cur['A'] < len(listA) and cur['B'] < len(listB):
            sim = jaro_winkler(listA[cur['A']], listB[cur['B']])
            if sim > 0.5:
                base['A'].append(listA[cur['A']])
                base['B'].append(listA[cur['B']])
                cur['A'] += 1
                cur['B'] += 1
            else:
                if listA[cur['A']] < listB[cur['B']]:
                    mis['B'].append(listB[cur['B']])
                    cur['B'] += 1
                else:
                    mis['A'].append(listB[cur['A']])
                    cur['A'] += 1

        if cur['A'] < len(listA):
            mis['A'].extend(listA[cur['A'] + 1:])
        if cur['B'] < len(listB):
            mis['B'].extend(listB[cur['B'] + 1:])

        return base, mis


class baseMetrics:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, size, njobs=2, accuracyresults=False):
        self.num_true_predicted_true = [0.0] * size
        self.num_true_predicted_false = [0.0] * size
        self.num_false_predicted_true = [0.0] * size
        self.num_false_predicted_false = [0.0] * size
        self.num_true = 0.0
        self.num_false = 0.0

        self.timer = 0.0
        self.timers = [0.0] * size
        self.result = {}
        self.file = None
        self.accuracyresults = accuracyresults
        if self.accuracyresults:
            self.file = open('dataset-accuracyresults-sim-metrics.txt', 'w+')

        self.predictedState = {
            'num_true_predicted_true': self.num_true_predicted_true,
            'num_true_predicted_false': self.num_true_predicted_false,
            'num_false_predicted_true': self.num_false_predicted_true,
            'num_false_predicted_false': self.num_false_predicted_false
        }
        self.njobs = njobs

    def __del__(self):
        if self.accuracyresults:
            self.file.close()

    def preprocessing(self, row):
        if row['res'] == "TRUE": self.num_true += 1.0
        else: self.num_false += 1.0

    def transform(self, strA, strB, sorting=False, stemming=False):
        a = strA
        b = strB

        # print("{0} - norm: {1}".format(row['s1'], normalize_str(row['s1'])))
        if sorting:
            a = " ".join(sorted_nicely(a.split(" ")))
            b = " ".join(sorted_nicely(b.split(" ")))
        if stemming:
            a = perform_stemming(a)
            b = perform_stemming(b)
        a = a.decode('utf-8')
        b = b.decode('utf-8')
        return a, b

    @abstractmethod
    def evaluate(self, row, sorting=False, stemming=False, permuted=False, freqTerms=None):
        pass

    @abstractmethod
    def print_stats(self):
        pass

    def prediction(self, sim_id, pred_val, real_val):
        result = ""
        var_name = ""
        if real_val == 1.0:
            if pred_val >= StaticValues.methods[sim_id - 1][1]:
                var_name = 'num_true_predicted_true'
                result = "\tTRUE"
            else:
                var_name = 'num_true_predicted_false'
                result = "\tFALSE"
        else:
            if pred_val >= StaticValues.methods[sim_id - 1][1]:
                var_name = 'num_false_predicted_true'
                result = "\tTRUE"
            else:
                var_name = 'num_false_predicted_false'
                result = "\tFALSE"

        return result, var_name


class calcSotAMetrics(baseMetrics):
    def __init__(self, njobs, accures):
        super(calcSotAMetrics, self).__init__(len(StaticValues.methods), njobs, accures)

    def generic_evaluator(self, idx, algnm, str1, str2, match):
        start_time = time.time()
        sim = StaticValues.algorithms[algnm](str1, str2)
        res, varnm = self.prediction(idx, sim, match)
        self.timers[idx - 1] += (time.time() - start_time)
        self.predictedState[varnm][idx - 1] += 1.0
        return res

    def evaluate(self, row, sorting=False, stemming=False, permuted=False, freqTerms=None):
        tot_res = ""
        real = 1.0 if row['res'] == "TRUE" else 0.0

        row['s1'], row['s2'] = self.transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming)

        tot_res += self.generic_evaluator(1, 'damerau_levenshtein', row['s1'], row['s2'], real)
        tot_res += self.generic_evaluator(8, 'jaccard', row['s1'], row['s2'], real)
        tot_res += self.generic_evaluator(2, 'jaro', row['s1'], row['s2'], real)
        tot_res += self.generic_evaluator(3, 'jaro_winkler', row['s1'], row['s2'], real)
        tot_res += self.generic_evaluator(4, 'jaro_winkler', row['s1'][::-1], row['s2'][::-1], real)
        tot_res += self.generic_evaluator(11, 'monge_elkan', row['s1'], row['s2'], real)
        tot_res += self.generic_evaluator(7, 'cosine', row['s1'], row['s2'], real)
        tot_res += self.generic_evaluator(9, 'strike_a_match', row['s1'], row['s2'], real)
        tot_res += self.generic_evaluator(12, 'soft_jaccard', row['s1'], row['s2'], real)
        tot_res += self.generic_evaluator(5, 'sorted_winkler', row['s1'], row['s2'], real)
        if permuted:
            tot_res += self.generic_evaluator(6, 'permuted_winkler', row['s1'], row['s2'], real)
        tot_res += self.generic_evaluator(10, 'skipgram', row['s1'], row['s2'], real)
        tot_res += self.generic_evaluator(13, 'davies', row['s1'], row['s2'], real)

        if self.accuracyresults:
            if real == 1.0:
                self.file.write("TRUE{0}".format(tot_res + "\n"))
            else:
                self.file.write("FALSE{0}".format(tot_res + "\n"))

    def print_stats(self):
        for idx in range(len(StaticValues.methods)):
            try:
                timer = (self.timers[idx] / float(int(self.num_true + self.num_false))) * 50000.0
                acc = (self.num_true_predicted_true[idx] + self.num_false_predicted_false[idx]) / \
                      (self.num_true + self.num_false)
                pre = (self.num_true_predicted_true[idx]) / \
                      (self.num_true_predicted_true[idx] + self.num_false_predicted_true[idx])
                rec = (self.num_true_predicted_true[idx]) / \
                      (self.num_true_predicted_true[idx] + self.num_true_predicted_false[idx])
                f1 = 2.0 * ((pre * rec) / (pre + rec))

                print "Metric = Supervised Classifier :", StaticValues.methods[idx][0]
                print "Accuracy =", acc
                print "Precision =", pre
                print "Recall =", rec
                print "F1 =", f1
                print "Processing time per 50K records =", timer
                print ""
                print "| Method\t\t& Accuracy\t& Precision\t& Recall\t& F1-Score\t& Time (50K Pairs)"
                print "||{0}\t& {1}\t& {2}\t& {3}\t& {4}\t& {5}".format(StaticValues.methods[idx][0], acc, pre, rec, f1, timer)
                print ""
                sys.stdout.flush()
            except ZeroDivisionError:
                pass
                # print "{0} is divided by zero\n".format(idx)

        # if results:
        #     return self.result


class calcCustomFEML(baseMetrics):
    names = [
        "Linear SVM", # "Gaussian Process",
        "Decision Tree", "Random Forest", "Neural Net", "AdaBoost", "Naive Bayes", "QDA", "LDA",
        "ExtraTreeClassifier", "XGBOOST"
    ]

    def __init__(self, njobs, accures):
        self.X1 = []
        self.Y1 = []
        self.X2 = []
        self.Y2 = []
        self.scores = []
        self.importances = []
        self.classifiers = [
            LinearSVC(random_state=0, C=1.0),
            # GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs=3, warm_start=True),
            DecisionTreeClassifier(random_state=0, max_depth=100, max_features='auto'),
            RandomForestClassifier(n_estimators=600, random_state=0, n_jobs=int(njobs), max_depth=100),
            MLPClassifier(alpha=1, random_state=0),
            AdaBoostClassifier(DecisionTreeClassifier(max_depth=100), n_estimators=600, random_state=0),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(), LinearDiscriminantAnalysis(),
            ExtraTreesClassifier(n_estimators=600, random_state=0, n_jobs=int(njobs), max_depth=100),
            XGBClassifier(n_estimators=3000, seed=0, nthread=int(njobs)),
        ]
        super(calcCustomFEML, self).__init__(len(self.classifiers), njobs, accures)

    def evaluate(self, row, sorting=False, stemming=False, permuted=False, freqTerms=False):
        if row['res'] == "TRUE":
            if len(self.Y1) < ((self.num_true + self.num_false) / 2.0): self.Y1.append(1.0)
            else: self.Y2.append(1.0)
        else:
            if len(self.Y1) < ((self.num_true + self.num_false) / 2.0): self.Y1.append(0.0)
            else: self.Y2.append(0.0)

        row['s1'], row['s2'] = self.transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming)

        start_time = time.time()
        sim1 = damerau_levenshtein(row['s1'], row['s2'])
        sim8 = jaccard(row['s1'], row['s2'])
        sim2 = jaro(row['s1'], row['s2'])
        sim3 = jaro_winkler(row['s1'], row['s2'])
        sim4 = jaro_winkler(row['s1'][::-1], row['s2'][::-1])
        sim11 = monge_elkan(row['s1'], row['s2'])
        sim7 = cosine(row['s1'], row['s2'])
        sim9 = strike_a_match(row['s1'], row['s2'])
        sim12 = soft_jaccard(row['s1'], row['s2'])
        sim5 = sorted_winkler(row['s1'], row['s2'])
        if permuted: sim6 = permuted_winkler(row['s1'], row['s2'])
        sim10 = skipgram(row['s1'], row['s2'])
        sim13 = davies(row['s1'], row['s2'])
        self.timer += (time.time() - start_time)
        if permuted:
            if len(self.X1) < ((self.num_true + self.num_false) / 2.0):
                self.X1.append([sim1, sim2, sim3, sim4, sim5, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            else: self.X2.append([sim1, sim2, sim3, sim4, sim5, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
        else:
            if len(self.X1) < ((self.num_true + self.num_false) / 2.0):
                self.X1.append([sim1, sim2, sim3, sim4, sim5, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            else: self.X2.append([sim1, sim2, sim3, sim4, sim5, sim7, sim8, sim9, sim10, sim11, sim12, sim13])

    def train_classifiers(self, polynomial=False):
        if polynomial:
            self.X1 = preprocessing.PolynomialFeatures().fit_transform(self.X1)
            self.X2 = preprocessing.PolynomialFeatures().fit_transform(self.X2)

        # iterate over classifiers
        for i, (name, clf) in enumerate(zip(self.names, self.classifiers)):
            scoreL = []
            importances = None
            print "Training {}...".format(name)

            clf.fit(np.array(self.X1), np.array(self.Y1))
            start_time = time.time()
            predictedL = list(clf.predict(np.array(self.X2)))
            self.timers[i] += (time.time() - start_time)
            if hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_
            elif hasattr(clf, "coef_"):
                importances = clf.coef_.ravel()
            scoreL.append(clf.score(np.array(self.X2), np.array(self.Y2)))

            clf.fit(np.array(self.X2), np.array(self.Y2))
            start_time = time.time()
            predictedL += list(clf.predict(np.array(self.X1)))
            self.timers[i] += (time.time() - start_time)
            if hasattr(clf, "feature_importances_"):
                importances += clf.feature_importances_
            elif hasattr(clf, "coef_"):
                # TODO when coef_ is added to importances that already contains another one, it throws a
                # ValueError: output array is read-only
                importances = clf.coef_.ravel()
            scoreL.append(clf.score(np.array(self.X1), np.array(self.Y1)))

            self.timers[i] += self.timer
            self.importances.append(importances)
            self.scores.append(scoreL)

            print "Matching records..."
            real = self.Y2 + self.Y1
            for pos in range(len(real)):
                if real[pos] == 1.0:
                    if predictedL[pos] == 1.0:
                        self.num_true_predicted_true[i] += 1.0
                        if self.accuracyresults:
                            self.file.write("TRUE\tTRUE\n")
                    else:
                        self.num_true_predicted_false[i] += 1.0
                        if self.accuracyresults:
                            self.file.write("TRUE\tFALSE\n")
                else:
                    if predictedL[pos] == 1.0:
                        self.num_false_predicted_true[i] += 1.0
                        if self.accuracyresults:
                            self.file.write("FALSE\tTRUE\n")
                    else:
                        self.num_false_predicted_false[i] += 1.0
                        if self.accuracyresults:
                            self.file.write("FALSE\tFALSE\n")

            # if hasattr(clf, "decision_function"):
            #     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            # else:
            #     Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    def print_stats(self):
        for idx, name in enumerate(self.names):
            try:
                timer = (self.timers[idx] / float(int(self.num_true + self.num_false))) * 50000.0
                acc = (self.num_true_predicted_true[idx] + self.num_false_predicted_false[idx]) / \
                      (self.num_true + self.num_false)
                pre = (self.num_true_predicted_true[idx]) / \
                      (self.num_true_predicted_true[idx] + self.num_false_predicted_true[idx])
                rec = (self.num_true_predicted_true[idx]) / \
                      (self.num_true_predicted_true[idx] + self.num_true_predicted_false[idx])
                f1 = 2.0 * ( ( pre * rec ) / ( pre + rec ) )

                print "Metric = Supervised Classifier :" , name
                print "Score (X2, X1) = ", self.scores[idx][0], self.scores[idx][1]
                print "Accuracy =", acc
                print "Precision =", pre
                print "Recall =", rec
                print "F1 =", f1
                print "Processing time per 50K records =", timer
                print "Number of training instances =", min(len(self.Y1), len(self.Y2))
                print ""
                print "| Method\t\t& Accuracy\t& Precision\t& Recall\t& F1-Score\t& Time (50K Pairs)"
                print "||{0}\t& {1}\t& {2}\t& {3}\t& {4}\t& {5}".format(name, acc, pre, rec, f1, timer)
                print ""
                sys.stdout.flush()

                try:
                    importances = self.importances[idx] / 2.0
                    indices = np.argsort(importances)[::-1]
                    for f in range(importances.shape[0]):
                        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
                except TypeError:
                    print "The classifier {} does not expose \"coef_\" or \"feature_importances_\" attributes".format(name)

                # if hasattr(clf, "feature_importances_"):
                #         # if results:
                #         #     result[indices[f]] = importances[indices[f]]
                print ""
                sys.stdout.flush()
            except ZeroDivisionError:
                pass

        # if results:
        #     return self.result


class calcDLearning(baseMetrics):
    pass


class calcSotAML(baseMetrics):
    pass


"""
Compute the Damerau-Levenshtein distance between two given
strings (s1 and s2)
https://www.guyrutenberg.com/2008/12/15/damerau-levenshtein-distance-in-python/
"""
# def damerau_levenshtein_distance(s1, s2):
#     d = {}
#     lenstr1 = len(s1)
#     lenstr2 = len(s2)
#     for i in xrange(-1, lenstr1 + 1):
#         d[(i, -1)] = i + 1
#     for j in xrange(-1, lenstr2 + 1):
#         d[(-1, j)] = j + 1
#
#     for i in xrange(lenstr1):
#         for j in xrange(lenstr2):
#             if s1[i] == s2[j]:
#                 cost = 0
#             else:
#                 cost = 1
#             d[(i, j)] = min(
#                 d[(i - 1, j)] + 1,  # deletion
#                 d[(i, j - 1)] + 1,  # insertion
#                 d[(i - 1, j - 1)] + cost,  # substitution
#             )
#             if i and j and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
#                 d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)  # transposition
#
#     return d[lenstr1 - 1, lenstr2 - 1]
