# -*- coding: utf-8 -*-

import os, sys
import time
from abc import ABCMeta, abstractmethod
import math
import re
import itertools
import unicodedata

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
from external.datasetcreator import strip_accents
from helpers import perform_stemming, normalize_str, sorted_nicely


def transform(strA, strB, sorting=False, stemming=False, canonical=False, delimiter=' '):
    a = strA.decode('utf8').lower()
    b = strB.decode('utf8').lower()

    if canonical:
        # NFKD: first applies a canonical decomposition, i.e., translates each character into its decomposed form.
        # and afterwards apply the compatibility decomposition, i.e. replace all compatibility characters with their
        # equivalents.
        # regex = re.compile(u'[‘’“”\'"!?;/⧸⁄‹›«»`]')

        # a = unicodedata.normalize('NFKD', a.decode('utf8')) # .encode('ASCII', 'ignore')
        # a = regex.sub('', a)
        a = strip_accents(a)
        b = strip_accents(b)

    if sorting:
        a = " ".join(sorted_nicely(a.split(delimiter)))
        b = " ".join(sorted_nicely(b.split(delimiter)))

    if stemming:
        a = perform_stemming(a)
        b = perform_stemming(b)
    return a, b


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

    nameIDs = {
        'damerau_levenshtein': 0,
        'davies': 12,
        'skipgram': 9,
        'permuted_winkler': 5,
        'sorted_winkler': 4,
        'soft_jaccard': 11,
        'strike_a_match': 8,
        'cosine': 6,
        'monge_elkan': 10,
        'jaro_winkler': 2,
        'jaro': 1,
        'jaccard': 7,
    }


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

    def _generic_metric_cmp(self, funcnm, a, b, sorting, stemming, canonical, invert=False):
        res = None

        tmp_a, tmp_b = transform(a, b)
        if invert:
            tmp_a = tmp_a[::-1]
            tmp_b = tmp_b[::-1]
        sim = StaticValues.algorithms[funcnm](tmp_a, tmp_b)

        if sorting:
            tmp_a, tmp_b = transform(a, b, sorting, stemming, canonical)
            if invert:
                tmp_a = tmp_a[::-1]
                tmp_b = tmp_b[::-1]
            if (sim - StaticValues.algorithms[funcnm](tmp_a, tmp_b)) > 0.05:
                    # and sim >= StaticValues.methods[StaticValues.nameIDs[funcnm]][1]:
                res = False
            else: res = True
        return res

    def cmp_score_after_transformation(self, row, sorting=False, stemming=False, canonical=False):
        sims_correlation = []

        res = self._generic_metric_cmp('damerau_levenshtein', row['s1'], row['s2'], sorting, stemming, canonical)
        if res is not None: sims_correlation.append(res)
        res = self._generic_metric_cmp('jaccard', row['s1'], row['s2'], sorting, stemming, canonical)
        if res is not None: sims_correlation.append(res)
        res = self._generic_metric_cmp('jaro', row['s1'], row['s2'], sorting, stemming, canonical)
        if res is not None: sims_correlation.append(res)
        res = self._generic_metric_cmp('jaro_winkler', row['s1'], row['s2'], sorting, stemming, canonical)
        if res is not None: sims_correlation.append(res)
        res = self._generic_metric_cmp('jaro_winkler', row['s1'], row['s2'], sorting, stemming, canonical, invert=True)
        if res is not None: sims_correlation.append(res)
        res = self._generic_metric_cmp('monge_elkan', row['s1'], row['s2'], sorting, stemming, canonical)
        if res is not None: sims_correlation.append(res)
        res = self._generic_metric_cmp('cosine', row['s1'], row['s2'], sorting, stemming, canonical)
        if res is not None: sims_correlation.append(res)
        res = self._generic_metric_cmp('strike_a_match', row['s1'], row['s2'], sorting, stemming, canonical)
        if res is not None: sims_correlation.append(res)
        res = self._generic_metric_cmp('soft_jaccard', row['s1'], row['s2'], sorting, stemming, canonical)
        if res is not None: sims_correlation.append(res)
        res = self._generic_metric_cmp('skipgram', row['s1'], row['s2'], sorting, stemming, canonical)
        if res is not None: sims_correlation.append(res)
        res = self._generic_metric_cmp('davies', row['s1'], row['s2'], sorting, stemming, canonical)
        if res is not None: sims_correlation.append(res)

        if sum(sims_correlation) > (len(sims_correlation) / 2.0): return True
        else: return False


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

    @abstractmethod
    def evaluate(self, row, sorting=False, stemming=False, canonical=False, permuted=False, freqTerms=None):
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

    def _generic_evaluator(self, idx, algnm, str1, str2, match):
        start_time = time.time()
        sim = StaticValues.algorithms[algnm](str1, str2)
        res, varnm = self.prediction(idx, sim, match)
        self.timers[idx - 1] += (time.time() - start_time)
        self.predictedState[varnm][idx - 1] += 1.0
        return res

    def evaluate(self, row, sorting=False, stemming=False, canonical=False, permuted=False, freqTerms=None):
        tot_res = ""
        real = 1.0 if row['res'] == "TRUE" else 0.0

        row['s1'], row['s2'] = transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming, canonical=canonical)

        tot_res += self._generic_evaluator(1, 'damerau_levenshtein', row['s1'], row['s2'], real)
        tot_res += self._generic_evaluator(8, 'jaccard', row['s1'], row['s2'], real)
        tot_res += self._generic_evaluator(2, 'jaro', row['s1'], row['s2'], real)
        tot_res += self._generic_evaluator(3, 'jaro_winkler', row['s1'], row['s2'], real)
        tot_res += self._generic_evaluator(4, 'jaro_winkler', row['s1'][::-1], row['s2'][::-1], real)
        tot_res += self._generic_evaluator(11, 'monge_elkan', row['s1'], row['s2'], real)
        tot_res += self._generic_evaluator(7, 'cosine', row['s1'], row['s2'], real)
        tot_res += self._generic_evaluator(9, 'strike_a_match', row['s1'], row['s2'], real)
        tot_res += self._generic_evaluator(12, 'soft_jaccard', row['s1'], row['s2'], real)
        tot_res += self._generic_evaluator(5, 'sorted_winkler', row['s1'], row['s2'], real)
        if permuted:
            tot_res += self._generic_evaluator(6, 'permuted_winkler', row['s1'], row['s2'], real)
        tot_res += self._generic_evaluator(10, 'skipgram', row['s1'], row['s2'], real)
        tot_res += self._generic_evaluator(13, 'davies', row['s1'], row['s2'], real)

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
    abbr_names = {
        'lsvm': 0,
        'dt': 1,
        'rf': 2,
        'nn': 3,
        'ada': 4,
        'nb': 5,
        'qda': 6,
        'lda': 7,
        'et': 8,
        'xgboost': 9,
    }

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
        self.scores = [[] for _ in xrange(len(self.classifiers))]
        self.importances = [0.0 for _ in xrange(len(self.classifiers))]
        self.mlalgs_to_run = self.abbr_names.keys()

        super(calcCustomFEML, self).__init__(len(self.classifiers), njobs, accures)

    def evaluate(self, row, sorting=False, stemming=False, canonical=False, permuted=False, freqTerms=False):
        if row['res'] == "TRUE":
            if len(self.Y1) < ((self.num_true + self.num_false) / 2.0): self.Y1.append(1.0)
            else: self.Y2.append(1.0)
        else:
            if len(self.Y1) < ((self.num_true + self.num_false) / 2.0): self.Y1.append(0.0)
            else: self.Y2.append(0.0)

        row['s1'], row['s2'] = transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming, canonical=canonical)

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

    def train_classifiers(self, ml_algs, polynomial=False):
        if polynomial:
            self.X1 = preprocessing.PolynomialFeatures().fit_transform(self.X1)
            self.X2 = preprocessing.PolynomialFeatures().fit_transform(self.X2)

        # iterate over classifiers
        if set(ml_algs) != {'all'}: self.mlalgs_to_run = ml_algs
        # for i, (name, clf) in enumerate(zip(self.names, self.classifiers)):
        for name in self.mlalgs_to_run:
            if name not in self.abbr_names.keys():
                print '{} is not a valid ML algorithm'.format(name)
                continue

            i = self.abbr_names[name]

            train_time = 0
            predictedL = list()
            print "Training {}...".format(self.names[i])
            for train_X, train_Y, pred_X, pred_Y in zip(
                    [row for row in [self.X1, self.X2]], [row for row in [self.Y1, self.Y2]],
                    [row for row in [self.X2, self.X1]], [row for row in [self.Y2, self.Y1]]
            ):
                start_time = time.time()
                self.classifiers[i].fit(np.array(train_X), np.array(train_Y))
                train_time = time.time() - start_time

                start_time = time.time()
                predictedL += list(self.classifiers[i].predict(np.array(pred_X)))
                self.timers[i] += (time.time() - start_time)

                if hasattr(self.classifiers[i], "feature_importances_"):
                    print self.classifiers[i].feature_importances_
                    self.importances[i] += self.classifiers[i].feature_importances_
                elif hasattr(self.classifiers[i], "coef_"):
                    self.importances[i] += self.classifiers[i].coef_.ravel()
                self.scores[i].append(self.classifiers[i].score(np.array(pred_X), np.array(pred_Y)))

            print "Training took {:.3f} min".format(train_time / 60.0)
            self.timers[i] += self.timer

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
        for name in self.mlalgs_to_run:
            if name not in self.abbr_names.keys():
                continue

            idx = self.abbr_names[name]
            try:
                timer = (self.timers[idx] / float(int(self.num_true + self.num_false))) * 50000.0
                acc = (self.num_true_predicted_true[idx] + self.num_false_predicted_false[idx]) / \
                      (self.num_true + self.num_false)
                pre = (self.num_true_predicted_true[idx]) / \
                      (self.num_true_predicted_true[idx] + self.num_false_predicted_true[idx])
                rec = (self.num_true_predicted_true[idx]) / \
                      (self.num_true_predicted_true[idx] + self.num_true_predicted_false[idx])
                f1 = 2.0 * ( ( pre * rec ) / ( pre + rec ) )

                print "Metric = Supervised Classifier :" , self.names[idx]
                print "Score (X2, X1) = ", self.scores[idx][0], self.scores[idx][1]
                print "Accuracy =", acc
                print "Precision =", pre
                print "Recall =", rec
                print "F1 =", f1
                print "Processing time per 50K records =", timer
                print "Number of training instances =", min(len(self.Y1), len(self.Y2))
                print ""
                print "| Method\t\t& Accuracy\t& Precision\t& Recall\t& F1-Score\t& Time (50K Pairs)"
                print "||{0}\t& {1}\t& {2}\t& {3}\t& {4}\t& {5}".format(self.names[idx], acc, pre, rec, f1, timer)
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
