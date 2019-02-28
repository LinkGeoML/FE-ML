# -*- coding: utf-8 -*-

# identifying str and unicode on Python 2, or str on Python 3
from six import string_types
import os, sys
import time
from abc import ABCMeta, abstractmethod
import re
import itertools
import glob
import csv

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from xgboost import XGBClassifier

from external.datasetcreator import strip_accents, LSimilarityVars, lsimilarity_terms
from helpers import perform_stemming, normalize_str, sorted_nicely, StaticValues


def transform(strA, strB, sorting=False, stemming=False, canonical=False, delimiter=' ', thres=0.6, only_sorting=False):
    a = strA.decode('utf8') #.lower()
    b = strB.decode('utf8') #.lower()

    if canonical:
        # NFKD: first applies a canonical decomposition, i.e., translates each character into its decomposed form.
        # and afterwards apply the compatibility decomposition, i.e. replace all compatibility characters with their
        # equivalents.

        # a = unicodedata.normalize('NFKD', a.decode('utf8')) # .encode('ASCII', 'ignore')
        a = strip_accents(a.lower())
        b = strip_accents(b.lower())

        regex = re.compile(u'[‘’“”\'"!?;/⧸⁄‹›«»`ʿ,.-]')
        a = regex.sub('', a)
        b = regex.sub('', b)

        # replace dashes with space
        # a = a.replace('-', ' ')
        # b = b.replace('-', ' ')

    if sorting:
        tmp_a = a.replace(' ', '')
        tmp_b = b.replace(' ', '')

        if StaticValues.algorithms['damerau_levenshtein'](tmp_a, tmp_b) <= thres:
            a = " ".join(sorted_nicely(a.split(delimiter)))
            b = " ".join(sorted_nicely(b.split(delimiter)))
        elif StaticValues.algorithms['damerau_levenshtein'](tmp_a, tmp_b) > StaticValues.algorithms['damerau_levenshtein'](a, b):
            a = tmp_a
            b = tmp_b
    elif only_sorting:
        a = " ".join(sorted_nicely(a.split(delimiter)))
        b = " ".join(sorted_nicely(b.split(delimiter)))

    if stemming:
        a = perform_stemming(a)
        b = perform_stemming(b)

    return a, b


def transform_str(str, stemming=False, canonical=False, delimiter=' '):
    a = str.decode('utf8')

    if canonical:
        # NFKD: first applies a canonical decomposition, i.e., translates each character into its decomposed form.
        # and afterwards apply the compatibility decomposition, i.e. replace all compatibility characters with their
        # equivalents.
        a = strip_accents(a.lower())

        regex = re.compile(u'[‘’“”\'"!?;/⧸⁄‹›«»`ʿ,.-]')
        a = regex.sub('', a)

    if stemming:
        a = perform_stemming(a)

    return a


class FEMLFeatures:
    no_freq_terms = 100

    def __init__(self):
        pass

    # TODO to_be_removed = "()/.,:!'"  # check the list of chars
    # Returned vals: #1: str1 is subset of str2, #2 str2 is subset of str1
    @staticmethod
    def contains(str1, str2):
        return all(x in str2 for x in str1.split())

    @staticmethod
    def contains_freq_term(str, freqTerms=None):
        str, _ = normalize_str(str)
        return True if freqTerms is not None and str in freqTerms else False

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

    @staticmethod
    def containsDashConnected_words(str):
        """
        Hyphenated words are considered to be:
            * a number of word chars
            * followed by any number of: a single hyphen followed by word chars
        """
        is_dashed = re.search(r"\w+(?:-\w+)+", str)
        return False if is_dashed is None else True

    @staticmethod
    def no_of_words(str1, str2):
        # str, _ = normalize_str(str)
        return len(set(str1.split())), len(set(str2.split()))

    @staticmethod
    def containsFreqTerms(str1, str2):
        specialTerms = dict(a=[], b=[])
        # specialTerms['a'] = filter(lambda x: x in a, freq_terms)
        # specialTerms['b'] = filter(lambda x: x in b, freq_terms)
        for idx, x in enumerate(LSimilarityVars.freq_ngrams['tokens'] + LSimilarityVars.freq_ngrams['chars']):
            if x in str1: specialTerms['a'].append([idx, x])
            if x in str2: specialTerms['b'].append([idx, x])

        return specialTerms['a'], specialTerms['b']

    @staticmethod
    def ngram_tokens(tokens, ngram=1):
        if tokens < 1: tokens = 1
        return list(itertools.chain.from_iterable([[tokens[i:i + ngram] for i in range(len(tokens) - (ngram - 1))]]))

    def ngrams(str, ngram=1):
       pass

    def _check_size(self, s):
        if not len(s) == 3:
            raise ValueError('expected size 3, got %d' % len(s))

    def containsInPos(self, str1, str2):
        fvec_str1 = []
        fvec_str2 = []

        sep_step = int(round(len(str1) / 3.0))
        fvec_str1.extend(
            [StaticValues.algorithms['damerau_levenshtein'](str1[0:sep_step], str2),
             StaticValues.algorithms['damerau_levenshtein'](str1[sep_step:2*sep_step], str2),
             StaticValues.algorithms['damerau_levenshtein'](str1[2*sep_step:], str2)]
        )

        sep_step = int(round(len(str2) / 3.0))
        fvec_str2.extend(
            [StaticValues.algorithms['damerau_levenshtein'](str1, str2[0:sep_step]),
             StaticValues.algorithms['damerau_levenshtein'](str1, str2[sep_step:2*sep_step]),
             StaticValues.algorithms['damerau_levenshtein'](str1, str2[2*sep_step:])]
        )

        self._check_size(fvec_str1)
        self._check_size(fvec_str2)

        return fvec_str1, fvec_str2

    def get_freqterms(self):
        if not os.path.isdir(os.path.join(os.getcwd(), 'input/')):
            print("Folder ./input/ does not exist")
        else:
            for f in glob.iglob('./input/*gram*.csv'):
                gram_type = 'tokens' if 'token' in os.path.basename(os.path.normpath(f)) else 'chars'
                with open(f) as csvfile:
                    print("Loading frequent terms from file {}...".format(f))
                    reader = csv.DictReader(csvfile, fieldnames=["term", "no"], delimiter='\t')
                    _ = reader.fieldnames
                    # go to next line after header
                    next(reader)

                    for i, row in enumerate(reader):
                        if i >= FEMLFeatures.no_freq_terms:
                            break

                        LSimilarityVars.freq_ngrams[gram_type].append(row['term'].decode('utf8'))
            print('Frequent terms loaded.')

    def update_weights(self, w):
        if isinstance(w, tuple) and len(w) >= 3:
            del LSimilarityVars.lsimilarity_weights[:]
            LSimilarityVars.lsimilarity_weights.extend(w[:3])

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

        self.file = None
        self.accuracyresults = accuracyresults

        self.feml_features = FEMLFeatures()

        self.predictedState = {
            'num_true_predicted_true': self.num_true_predicted_true,
            'num_true_predicted_false': self.num_true_predicted_false,
            'num_false_predicted_true': self.num_false_predicted_true,
            'num_false_predicted_false': self.num_false_predicted_false
        }
        self.njobs = njobs

    def __del__(self):
        if self.accuracyresults and self.file is not None and not self.file.closed:
            self.file.close()

    def reset_vars(self):
        self.num_true_predicted_true[:] = [0.0] * len(self.num_true_predicted_true)
        self.num_true_predicted_false[:] = [0.0] * len(self.num_true_predicted_false)
        self.num_false_predicted_true[:] = [0.0] * len(self.num_false_predicted_true)
        self.num_false_predicted_false[:] = [0.0] * len(self.num_false_predicted_false)

        self.timer = 0.0
        self.timers[:] = [0.0] * len(self.timers)

    def preprocessing(self, row):
        if row['res'] == "TRUE": self.num_true += 1.0
        else: self.num_false += 1.0

    @abstractmethod
    def evaluate(self, row, sorting=False, stemming=False, canonical=False, permuted=False, freqTerms=None, custom_thres='orig'):
        pass

    def print_stats(self):
        for idx, m in enumerate(StaticValues.methods):
            try:
                timer = (self.timers[idx] / float(int(self.num_true + self.num_false))) * 50000.0
                acc = (self.num_true_predicted_true[idx] + self.num_false_predicted_false[idx]) / \
                      (self.num_true + self.num_false)
                pre = (self.num_true_predicted_true[idx]) / \
                      (self.num_true_predicted_true[idx] + self.num_false_predicted_true[idx])
                rec = (self.num_true_predicted_true[idx]) / \
                      (self.num_true_predicted_true[idx] + self.num_true_predicted_false[idx])
                f1 = 2.0 * ((pre * rec) / (pre + rec))

                print("Metric = Supervised Classifier :", m[0])
                print("Accuracy =", acc)
                print("Precision =", pre)
                print("Recall =", rec)
                print("F1 =", f1)
                print("Processing time per 50K records =", timer)
                print("")
                print("| Method\t\t& Accuracy\t& Precision\t& Recall\t& F1-Score\t& Time (50K Pairs)")
                print("||{0}\t& {1}\t& {2}\t& {3}\t& {4}\t& {5}".format(m[0], acc, pre, rec, f1, timer))
                print("")
                sys.stdout.flush()
            except ZeroDivisionError:
                pass
                # print "{0} is divided by zero\n".format(StaticValues.methods[idx][0])

        # if results:
        #     return self.result

    def get_stats(self):
        res = {}
        for idx, m in enumerate(StaticValues.methods):
            try:
                acc = (self.num_true_predicted_true[idx] + self.num_false_predicted_false[idx]) / \
                      (self.num_true + self.num_false)
                pre = (self.num_true_predicted_true[idx]) / \
                      (self.num_true_predicted_true[idx] + self.num_false_predicted_true[idx])
                rec = (self.num_true_predicted_true[idx]) / \
                      (self.num_true_predicted_true[idx] + self.num_true_predicted_false[idx])
                f1 = 2.0 * ((pre * rec) / (pre + rec))

                res[m[0]] = [acc, pre, rec, f1]
            except ZeroDivisionError:
                pass
                # print('zero division for method {}'.format(m[0]))

        return res

    def prediction(self, sim_id, pred_val, real_val, custom_thres):
        result = ""
        var_name = ""

        thres = StaticValues.methods[sim_id - 1][1][custom_thres] if isinstance(custom_thres, string_types) else custom_thres
        if real_val == 1.0:
            if pred_val >= thres:
                var_name = 'num_true_predicted_true'
                result = "\tTRUE"
            else:
                var_name = 'num_true_predicted_false'
                result = "\tFALSE"
        else:
            if pred_val >= thres:
                var_name = 'num_false_predicted_true'
                result = "\tTRUE"
            else:
                var_name = 'num_false_predicted_false'
                result = "\tFALSE"

        return result, var_name

    def freq_terms_list(self):
        self.feml_features.get_freqterms()


class calcSotAMetrics(baseMetrics):
    def __init__(self, njobs, accures):
        super(calcSotAMetrics, self).__init__(len(StaticValues.methods), njobs, accures)

    def _generic_evaluator(self, idx, algnm, str1, str2, is_a_match, custom_thres):
        start_time = time.time()
        sim_val = StaticValues.algorithms[algnm](str1, str2)
        res, varnm = self.prediction(idx, sim_val, is_a_match, custom_thres)
        self.timers[idx - 1] += (time.time() - start_time)
        self.predictedState[varnm][idx - 1] += 1.0
        return res

    def evaluate(self, row, sorting=False, stemming=False, canonical=False, permuted=False, freqTerms=None, custom_thres='orig'):
        tot_res = ""
        flag_true_match = 1.0 if row['res'] == "TRUE" else 0.0

        a, b = transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming, canonical=canonical)

        tot_res += self._generic_evaluator(1, 'damerau_levenshtein', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(8, 'jaccard', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(2, 'jaro', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(3, 'jaro_winkler', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(4, 'jaro_winkler', a[::-1], b[::-1], flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(11, 'monge_elkan', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(7, 'cosine', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(9, 'strike_a_match', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(12, 'soft_jaccard', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(5, 'sorted_winkler', a, b, flag_true_match, custom_thres)
        if permuted: tot_res += self._generic_evaluator(6, 'permuted_winkler', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(10, 'skipgram', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(13, 'davies', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(14, 'l_jaro_winkler', a, b, flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(15, 'l_jaro_winkler', a[::-1], b[::-1], flag_true_match, custom_thres)
        tot_res += self._generic_evaluator(16, 'lsimilarity', a, b, flag_true_match, custom_thres)

        if self.accuracyresults:
            if self.file is None:
                file_name = 'dataset-accuracyresults-sim-metrics'
                if canonical:
                    file_name += '_canonical'
                if sorting:
                    file_name += '_sorted'
                self.file = open(file_name + '.csv', 'w+')

            if flag_true_match == 1.0:
                self.file.write("TRUE{0}\t{1}\t{2}\n".format(tot_res, a.encode('utf8'), b.encode('utf8')))
            else:
                self.file.write("FALSE{0}\t{1}\t{2}\n".format(tot_res, a.encode('utf8'), b.encode('utf8')))

    def evaluate_sorting(self, row, custom_thres, stemming=False, permuted=False):
        tot_res = ""
        flag_true_match = 1.0 if row['res'] == "TRUE" else 0.0

        row['s1'], row['s2'] = transform(row['s1'], row['s2'], sorting=True, stemming=stemming, canonical=True, thres=custom_thres)

        tot_res += self._generic_evaluator(1, 'damerau_levenshtein', row['s1'], row['s2'], flag_true_match, 'sorted')
        tot_res += self._generic_evaluator(8, 'jaccard', row['s1'], row['s2'], flag_true_match, 'sorted')
        tot_res += self._generic_evaluator(2, 'jaro', row['s1'], row['s2'], flag_true_match, 'sorted')
        tot_res += self._generic_evaluator(3, 'jaro_winkler', row['s1'], row['s2'], flag_true_match, 'sorted')
        tot_res += self._generic_evaluator(4, 'jaro_winkler', row['s1'][::-1], row['s2'][::-1], flag_true_match, 'sorted')
        tot_res += self._generic_evaluator(11, 'monge_elkan', row['s1'], row['s2'], flag_true_match, 'sorted')
        tot_res += self._generic_evaluator(7, 'cosine', row['s1'], row['s2'], flag_true_match, 'sorted')
        tot_res += self._generic_evaluator(9, 'strike_a_match', row['s1'], row['s2'], flag_true_match, 'sorted')
        tot_res += self._generic_evaluator(12, 'soft_jaccard', row['s1'], row['s2'], flag_true_match, 'sorted')
        tot_res += self._generic_evaluator(5, 'sorted_winkler', row['s1'], row['s2'], flag_true_match, 'sorted')
        if permuted: tot_res += self._generic_evaluator(6, 'permuted_winkler', row['s1'], row['s2'], flag_true_match, 'sorted')
        tot_res += self._generic_evaluator(10, 'skipgram', row['s1'], row['s2'], flag_true_match, 'sorted')
        tot_res += self._generic_evaluator(13, 'davies', row['s1'], row['s2'], flag_true_match, 'sorted')

        if self.accuracyresults:
            if self.file is None:
                file_name = 'dataset-accuracyresults-sim-metrics'
                if True:
                    file_name += '_canonical'
                if True:
                    file_name += '_sorted'
                self.file = open(file_name + '.csv', 'w+')

            if flag_true_match == 1.0:
                self.file.write("TRUE{0}\t{1}\t{2}\n".format(tot_res, row['s1'].encode('utf8'), row['s2'].encode('utf8')))
            else:
                self.file.write("FALSE{0}\t{1}\t{2}\n".format(tot_res, row['s1'].encode('utf8'), row['s2'].encode('utf8')))


class calcCustomFEML(baseMetrics):
    max_important_features_toshow = 10

    def __init__(self, njobs, accures):
        self.X1 = []
        self.Y1 = []
        self.X2 = []
        self.Y2 = []

        self.classifiers = [
            LinearSVC(random_state=0, C=1.0),
            # GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs=3, warm_start=True),
            DecisionTreeClassifier(random_state=0, max_depth=50, max_features='auto'),
            RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=int(njobs), max_depth=50),
            MLPClassifier(alpha=1, random_state=0),
            # AdaBoostClassifier(DecisionTreeClassifier(max_depth=50), n_estimators=300, random_state=0),
            GaussianNB(),
            # QuadraticDiscriminantAnalysis(), LinearDiscriminantAnalysis(),
            ExtraTreesClassifier(n_estimators=150, random_state=0, n_jobs=int(njobs), max_depth=50),
            XGBClassifier(n_estimators=3000, seed=0, nthread=int(njobs)),
        ]
        self.scores = [[] for _ in range(len(self.classifiers))]
        self.importances = [0.0 for _ in range(len(self.classifiers))]
        self.mlalgs_to_run = StaticValues.classifiers_abbr.keys()

        super(calcCustomFEML, self).__init__(len(self.classifiers), njobs, accures)

    def evaluate(self, row, sorting=False, stemming=False, canonical=False, permuted=False, freqTerms=False, custom_thres='orig'):
        if row['res'] == "TRUE":
            if len(self.Y1) < ((self.num_true + self.num_false) / 2.0): self.Y1.append(1.0)
            else: self.Y2.append(1.0)
        else:
            if len(self.Y1) < ((self.num_true + self.num_false) / 2.0): self.Y1.append(0.0)
            else: self.Y2.append(0.0)

        row['s1'], row['s2'] = transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming, canonical=canonical)

        start_time = time.time()
        sim1 = StaticValues.algorithms['damerau_levenshtein'](row['s1'], row['s2'])
        sim8 = StaticValues.algorithms['jaccard'](row['s1'], row['s2'])
        sim2 = StaticValues.algorithms['jaro'](row['s1'], row['s2'])
        sim3 = StaticValues.algorithms['jaro_winkler'](row['s1'], row['s2'])
        sim4 = StaticValues.algorithms['jaro_winkler'](row['s1'][::-1], row['s2'][::-1])
        sim11 = StaticValues.algorithms['monge_elkan'](row['s1'], row['s2'])
        sim7 = StaticValues.algorithms['cosine'](row['s1'], row['s2'])
        sim9 = StaticValues.algorithms['strike_a_match'](row['s1'], row['s2'])
        sim12 = StaticValues.algorithms['soft_jaccard'](row['s1'], row['s2'])
        sim5 = StaticValues.algorithms['sorted_winkler'](row['s1'], row['s2'])
        if permuted: sim6 = StaticValues.algorithms['permuted_winkler'](row['s1'], row['s2'])
        sim10 = StaticValues.algorithms['skipgram'](row['s1'], row['s2'])
        sim13 = StaticValues.algorithms['davies'](row['s1'], row['s2'])
        self.timer += (time.time() - start_time)

        if len(self.X1) < ((self.num_true + self.num_false) / 2.0):
            if permuted:
                self.X1.append([sim1, sim2, sim3, sim4, sim5, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            else:
                self.X1.append([sim1, sim2, sim3, sim4, sim5, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
        else:
            if permuted:
                self.X2.append([sim1, sim2, sim3, sim4, sim5, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            else:
                self.X2.append([sim1, sim2, sim3, sim4, sim5, sim7, sim8, sim9, sim10, sim11, sim12, sim13])

        if self.file is None and self.accuracyresults:
            file_name = 'dataset-accuracyresults-sim-metrics'
            if canonical:
                file_name += '_canonical'
            if sorting:
                file_name += '_sorted'
            self.file = open(file_name + '.csv', 'w+')

    def train_classifiers(self, ml_algs, polynomial=False, standardize=False):
        if polynomial:
            self.X1 = PolynomialFeatures().fit_transform(self.X1)
            self.X2 = PolynomialFeatures().fit_transform(self.X2)

        # iterate over classifiers
        if set(ml_algs) != {'all'}: self.mlalgs_to_run = ml_algs
        # for i, (name, clf) in enumerate(zip(self.names, self.classifiers)):
        for name in self.mlalgs_to_run:
            if name not in StaticValues.classifiers_abbr.keys():
                print('{} is not a valid ML algorithm'.format(name))
                continue

            i = StaticValues.classifiers_abbr[name]
            model = self.classifiers[i]

            train_time = 0
            predictedL = list()
            print("Training {}...".format(StaticValues.classifiers[i]))
            for train_X, train_Y, pred_X, pred_Y in zip(
                    [row for row in [self.X1, self.X2]], [row for row in [self.Y1, self.Y2]],
                    [row for row in [self.X2, self.X1]], [row for row in [self.Y2, self.Y1]]
            ):
                start_time = time.time()
                model.fit(np.array(train_X), np.array(train_Y))
                train_time += (time.time() - start_time)

                start_time = time.time()
                predictedL += list(model.predict(np.array(pred_X)))
                self.timers[i] += (time.time() - start_time)

                if hasattr(model, "feature_importances_"):
                    self.importances[i] += model.feature_importances_
                elif hasattr(model, "coef_"):
                    self.importances[i] += model.coef_.ravel()
                self.scores[i].append(model.score(np.array(pred_X), np.array(pred_Y)))

            print("Training took {:.3f} min".format(train_time / (2 * 60.0)))
            self.timers[i] += self.timer

            print("Matching records...")
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
            if name not in StaticValues.classifiers_abbr.keys():
                continue

            idx = StaticValues.classifiers_abbr[name]
            try:
                timer = (self.timers[idx] / float(int(self.num_true + self.num_false))) * 50000.0
                acc = (self.num_true_predicted_true[idx] + self.num_false_predicted_false[idx]) / \
                      (self.num_true + self.num_false)
                pre = (self.num_true_predicted_true[idx]) / \
                      (self.num_true_predicted_true[idx] + self.num_false_predicted_true[idx])
                rec = (self.num_true_predicted_true[idx]) / \
                      (self.num_true_predicted_true[idx] + self.num_true_predicted_false[idx])
                f1 = 2.0 * ( ( pre * rec ) / ( pre + rec ) )

                print "Metric = Supervised Classifier :" , StaticValues.classifiers[idx]
                print "Score (X2, X1) = ", self.scores[idx][0], self.scores[idx][1]
                print "Accuracy =", acc
                print "Precision =", pre
                print "Recall =", rec
                print "F1 =", f1
                print "Processing time per 50K records =", timer
                print "Number of training instances =", min(len(self.Y1), len(self.Y2))
                print ""
                print "| Method\t\t& Accuracy\t& Precision\t& Recall\t& F1-Score\t& Time (50K Pairs)"
                print "||{0}\t& {1}\t& {2}\t& {3}\t& {4}\t& {5}".format(StaticValues.classifiers[idx], acc, pre, rec, f1, timer)
                print ""
                sys.stdout.flush()

                importances = self.importances[idx] / 2.0
                if isinstance(importances, float):
                    print "The classifier {} does not expose \"coef_\" or \"feature_importances_\" attributes".format(
                        name)
                else:
                    indices = np.argsort(importances)[::-1]
                    for f in range(min(importances.shape[0], self.max_important_features_toshow)):
                            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

                # if hasattr(clf, "feature_importances_"):
                #         # if results:
                #         #     result[indices[f]] = importances[indices[f]]
                print ""
                sys.stdout.flush()
            except ZeroDivisionError:
                pass

        # if results:
        #     return self.result


class calcCustomFEMLExtended(baseMetrics):
    max_important_features_toshow = 20
    fterm_feature_size = 20

    def __init__(self, njobs, accures):
        self.X1 = []
        self.Y1 = []
        self.X2 = []
        self.Y2 = []

        self.classifiers = [
            LinearSVC(random_state=0, C=1.0, max_iter=2000),
            # GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs=3, warm_start=True),
            DecisionTreeClassifier(random_state=0, max_depth=50, max_features='auto'),
            RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=int(njobs), max_depth=50),
            MLPClassifier(alpha=1, random_state=0),
            # AdaBoostClassifier(DecisionTreeClassifier(max_depth=50), n_estimators=300, random_state=0),
            GaussianNB(),
            # QuadraticDiscriminantAnalysis(), LinearDiscriminantAnalysis(),
            ExtraTreesClassifier(n_estimators=150, random_state=0, n_jobs=int(njobs), max_depth=50),
            XGBClassifier(n_estimators=3000, seed=0, nthread=int(njobs)),
        ]
        self.scores = [[] for _ in range(len(self.classifiers))]
        self.importances = [0.0 for _ in range(len(self.classifiers))]
        self.mlalgs_to_run = StaticValues.classifiers_abbr.keys()

        super(calcCustomFEMLExtended, self).__init__(len(self.classifiers), njobs, accures)

    def evaluate(self, row, sorting=False, stemming=False, canonical=False, permuted=False, freqTerms=False, custom_thres='orig'):
        if row['res'] == "TRUE":
            if len(self.Y1) < ((self.num_true + self.num_false) / 2.0): self.Y1.append(1.0)
            else: self.Y2.append(1.0)
        else:
            if len(self.Y1) < ((self.num_true + self.num_false) / 2.0): self.Y1.append(0.0)
            else: self.Y2.append(0.0)

        row['s1'], row['s2'] = transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming, canonical=canonical)

        start_time = time.time()
        sim1 = StaticValues.algorithms['damerau_levenshtein'](row['s1'], row['s2'])
        sim8 = StaticValues.algorithms['jaccard'](row['s1'], row['s2'])
        sim2 = StaticValues.algorithms['jaro'](row['s1'], row['s2'])
        sim3 = StaticValues.algorithms['jaro_winkler'](row['s1'], row['s2'])
        sim4 = StaticValues.algorithms['jaro_winkler'](row['s1'][::-1], row['s2'][::-1])
        sim11 = StaticValues.algorithms['monge_elkan'](row['s1'], row['s2'])
        sim7 = StaticValues.algorithms['cosine'](row['s1'], row['s2'])
        sim9 = StaticValues.algorithms['strike_a_match'](row['s1'], row['s2'])
        sim12 = StaticValues.algorithms['soft_jaccard'](row['s1'], row['s2'])
        sim5 = StaticValues.algorithms['sorted_winkler'](row['s1'], row['s2'])
        if permuted: sim6 = StaticValues.algorithms['permuted_winkler'](row['s1'], row['s2'])
        sim10 = StaticValues.algorithms['skipgram'](row['s1'], row['s2'])
        sim13 = StaticValues.algorithms['davies'](row['s1'], row['s2'])

        feature1_1, feature1_2, feature1_3 = lsimilarity_terms(row['s1'], row['s2'], 0.7)
        feature2_1 = FEMLFeatures.contains(row['s1'], row['s2'])
        feature2_2 = FEMLFeatures.contains(row['s2'], row['s1'])
        feature3_1, feature3_2 = FEMLFeatures.no_of_words(row['s1'], row['s2'])
        feature4_1 = FEMLFeatures.containsDashConnected_words(row['s1'])
        feature4_2 = FEMLFeatures.containsDashConnected_words(row['s2'])
        fterms_s1, fterms_s2 = FEMLFeatures.containsFreqTerms(row['s1'], row['s2'])
        feature5_1 = False if len(fterms_s1) == 0 else True
        feature5_2 = False if len(fterms_s2) == 0 else True
        feature6_1, feature6_2 = FEMLFeatures().containsInPos(row['s1'], row['s2'])
        feature7_1 = [0] * (len(LSimilarityVars.freq_ngrams['tokens']) + len(LSimilarityVars.freq_ngrams['chars']))
        feature7_2 = [0] * (len(LSimilarityVars.freq_ngrams['tokens']) + len(LSimilarityVars.freq_ngrams['chars']))
        for x in fterms_s1: feature7_1[x[0]] = 1
        for x in fterms_s2: feature7_2[x[0]] = 1

        self.timer += (time.time() - start_time)
        if len(self.X1) < ((self.num_true + self.num_false) / 2.0):
            if permuted:
                self.X1.append([sim1, sim2, sim3, sim4, sim5, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            else:
                self.X1.append([sim1, sim2, sim3, sim4, sim5, sim7, sim8, sim9, sim10, sim11, sim12, sim13])

            self.X1[-1].extend([
                feature1_1, feature1_2, feature1_3,
                int(feature2_1), int(feature2_2),
                feature3_1, feature3_2,
                int(feature4_1), int(feature4_2),
                int(feature5_1), int(feature5_2)
            ])
            self.X1[-1].extend(map(lambda x: int(x == max(feature6_1)), feature6_1))
            self.X1[-1].extend(map(lambda x: int(x == max(feature6_2)), feature6_2))
            self.X1[-1].extend(feature7_1[:self.fterm_feature_size] + feature7_2[:self.fterm_feature_size])
        else:
            if permuted:
                self.X2.append([sim1, sim2, sim3, sim4, sim5, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            else:
                self.X2.append([sim1, sim2, sim3, sim4, sim5, sim7, sim8, sim9, sim10, sim11, sim12, sim13])

            self.X2[-1].extend([
                feature1_1, feature1_2, feature1_3,
                int(feature2_1), int(feature2_2),
                feature3_1, feature3_2,
                int(feature4_1), int(feature4_2),
                int(feature5_1), int(feature5_2)
            ])
            self.X2[-1].extend(map(lambda x: int(x == max(feature6_1)), feature6_1))
            self.X2[-1].extend(map(lambda x: int(x == max(feature6_2)), feature6_2))
            self.X2[-1].extend(feature7_1[:self.fterm_feature_size] + feature7_2[:self.fterm_feature_size])

        if self.file is None and self.accuracyresults:
            file_name = 'dataset-accuracyresults-sim-metrics'
            if canonical:
                file_name += '_canonical'
            if sorting:
                file_name += '_sorted'
            self.file = open(file_name + '.csv', 'w+')

    def train_classifiers(self, ml_algs, polynomial=False, standardize=False):
        if polynomial:
            self.X1 = PolynomialFeatures().fit_transform(self.X1)
            self.X2 = PolynomialFeatures().fit_transform(self.X2)
        if standardize:
            # self.X1 = StandardScaler().fit_transform(self.X1)
            # self.X2 = StandardScaler().fit_transform(self.X2)
            # print(zip(*self.X1)[18][:10], '||', zip(*self.X2)[18][:10])
            self.X1 = MinMaxScaler().fit_transform(self.X1)
            self.X2 = MinMaxScaler().fit_transform(self.X2)
            # print(zip(*self.X1)[18][:10], '||', zip(*self.X2)[18][:10])

        # iterate over classifiers
        if set(ml_algs) != {'all'}: self.mlalgs_to_run = ml_algs
        # for i, (name, clf) in enumerate(zip(self.names, self.classifiers)):
        for name in self.mlalgs_to_run:
            if name not in StaticValues.classifiers_abbr.keys():
                print('{} is not a valid ML algorithm'.format(name))
                continue

            i = StaticValues.classifiers_abbr[name]
            model = self.classifiers[i]

            train_time = 0
            predictedL = list()
            print("Training {}...".format(StaticValues.classifiers[i]))
            for train_X, train_Y, pred_X, pred_Y in zip(
                    [row for row in [self.X1, self.X2]], [row for row in [self.Y1, self.Y2]],
                    [row for row in [self.X2, self.X1]], [row for row in [self.Y2, self.Y1]]
            ):
                start_time = time.time()
                model.fit(np.array(train_X), np.array(train_Y))
                train_time += (time.time() - start_time)

                start_time = time.time()
                predictedL += list(model.predict(np.array(pred_X)))
                self.timers[i] += (time.time() - start_time)

                if hasattr(model, "feature_importances_"):
                    self.importances[i] += model.feature_importances_
                elif hasattr(model, "coef_"):
                    self.importances[i] += model.coef_.ravel()
                self.scores[i].append(model.score(np.array(pred_X), np.array(pred_Y)))

            print("Training took {:.3f} min".format(train_time / (2 * 60.0)))
            self.timers[i] += self.timer

            print("Matching records...")
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
            if name not in StaticValues.classifiers_abbr.keys():
                continue

            idx = StaticValues.classifiers_abbr[name]
            try:
                timer = (self.timers[idx] / float(int(self.num_true + self.num_false))) * 50000.0
                acc = (self.num_true_predicted_true[idx] + self.num_false_predicted_false[idx]) / \
                      (self.num_true + self.num_false)
                pre = (self.num_true_predicted_true[idx]) / \
                      (self.num_true_predicted_true[idx] + self.num_false_predicted_true[idx])
                rec = (self.num_true_predicted_true[idx]) / \
                      (self.num_true_predicted_true[idx] + self.num_true_predicted_false[idx])
                f1 = 2.0 * ( ( pre * rec ) / ( pre + rec ) )

                print "Metric = Supervised Classifier :" , StaticValues.classifiers[idx]
                print "Score (X2, X1) = ", self.scores[idx][0], self.scores[idx][1]
                print "Accuracy =", acc
                print "Precision =", pre
                print "Recall =", rec
                print "F1 =", f1
                print "Processing time per 50K records =", timer
                print "Number of training instances =", min(len(self.Y1), len(self.Y2))
                print ""
                print "| Method\t\t& Accuracy\t& Precision\t& Recall\t& F1-Score\t& Time (50K Pairs)"
                print "||{0}\t& {1}\t& {2}\t& {3}\t& {4}\t& {5}".format(StaticValues.classifiers[idx], acc, pre, rec, f1, timer)
                print ""
                sys.stdout.flush()

                importances = self.importances[idx] / 2.0
                if isinstance(importances, float):
                    print "The classifier {} does not expose \"coef_\" or \"feature_importances_\" attributes".format(
                        name)
                else:
                    indices = np.argsort(importances)[::-1]
                    for f in range(min(importances.shape[0], self.max_important_features_toshow)):
                            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

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


class testMetrics(baseMetrics):
    def __init__(self, njobs, accures):
        super(testMetrics, self).__init__(len(StaticValues.methods), njobs, accures)

    def _generic_evaluator(self, idx, algnm, str1, str2, is_a_match, custom_thres):
        start_time = time.time()
        sim_val = StaticValues.algorithms[algnm](str1, str2)
        res, varnm = self.prediction(idx, sim_val, is_a_match, custom_thres)
        self.timers[idx - 1] += (time.time() - start_time)
        self.predictedState[varnm][idx - 1] += 1.0
        return res

    def evaluate(self, row, sorting=False, stemming=False, canonical=False, permuted=False, freqTerms=None, custom_thres='orig'):
        tot_res = ""
        flag_true_match = 1.0 if row['res'] == "TRUE" else 0.0

        a, b = transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming, canonical=canonical)
        tot_res += self._generic_evaluator(16, 'lsimilarity', a, b, flag_true_match, custom_thres)

        if self.accuracyresults:
            if self.file is None:
                file_name = 'dataset-accuracyresults-sim-metrics'
                if canonical:
                    file_name += '_canonical'
                if sorting:
                    file_name += '_sorted'
                self.file = open(file_name + '.csv', 'w+')

            if flag_true_match == 1.0:
                self.file.write("TRUE{0}\t{1}\t{2}\n".format(tot_res, a.encode('utf8'), b.encode('utf8')))
            else:
                self.file.write("FALSE{0}\t{1}\t{2}\n".format(tot_res, a.encode('utf8'), b.encode('utf8')))


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
