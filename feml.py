"""Feature extraction and traditional classifiers for toponym matching.

Usage:
  feml.py [options]
  feml.py (-h | --help)
  feml.py --version

Options:
  -h --help                 show this screen.
  --version                 show version.
  -c <classifier_method>    various supported classifiers. [default: rf].
  -d <dataset-name>         dataset to use. [default: dataset-string-similarity.txt]
  --permuted                Use permuted Jaro-Winkler metrics. Default is False.
  --stemming                Perform stemming. Default is False.
  --sort                    Sort alphanumerically.
  --ev <evaluator_type>     Type of experiments to conduct. [default: SotAMetrics]
  --print                   Print only computed variables. Default is False.
  --accuracyresults         Store predicted results (TRUE/FALSE) in file. Default is False

Arguments:
  classifier_method:        'rf' (default)
                            'et'
                            'svm'
                            'xgboost'
  evaluator_type            'SotAMetrics' (default)
                            'MLCustom'
                            'DL'
                            'SortedMetrics'

"""

import os, sys
import csv
import time
from collections import Counter
import re
from abc import ABCMeta, abstractmethod
import itertools
import math

# import configparser
from docopt import docopt
from nltk import SnowballStemmer, wordpunct_tokenize
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import stopwords
from langdetect import detect, lang_detect_exception
import pycountry

import helpers
from featureclassifiers import evaluate_classifier
from datasetcreator import damerau_levenshtein, jaccard, jaro, jaro_winkler,monge_elkan, cosine, strike_a_match, \
    soft_jaccard, sorted_winkler, permuted_winkler, skipgram, davies
from datasetcreator import detect_alphabet, fields


"""
Compute the Damerau-Levenshtein distance between two given
strings (s1 and s2)
https://www.guyrutenberg.com/2008/12/15/damerau-levenshtein-distance-in-python/
"""
def damerau_levenshtein_distance(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in xrange(-1, lenstr1 + 1):
        d[(i, -1)] = i + 1
    for j in xrange(-1, lenstr2 + 1):
        d[(-1, j)] = j + 1

    for i in xrange(lenstr1):
        for j in xrange(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,  # deletion
                d[(i, j - 1)] + 1,  # insertion
                d[(i - 1, j - 1)] + cost,  # substitution
            )
            if i and j and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)  # transposition

    return d[lenstr1 - 1, lenstr2 - 1]

def get_langnm(str, lang_detect=False):
    lname = 'english'
    try:
        lname = pycountry.languages.get(alpha_2=detect(str)).name.lower() if lang_detect else 'english'
    except lang_detect_exception.LangDetectException as e:
        print(e)

    return lname

# Clean the string from stopwords, puctuations based on language detections feature
# Returned values #1: non-stopped words, #2: stopped words
def normalize_str(str, sstopwords=None, sorted=False, lang_detect=False):
    languagenm = get_langnm(str) if lang_detect else 'english'

    tokens = wordpunct_tokenize(str)
    words = [word.lower() for word in tokens if word.isalpha()]
    stopwords_set = set(stopwords.words(languagenm)) if sstopwords is None else set(sstopwords)

    filtered_words = sorted_nicely(filter(lambda token: token not in stopwords_set, words)) if sorted else \
        filter(lambda token: token not in stopwords_set, words)
    stopped_words = sorted_nicely(filter(lambda token: token not in filtered_words, words)) if sorted else \
        filter(lambda token: token not in filtered_words, words)

    return filtered_words, stopped_words

def perform_stemming(str, lang_detect=False):
    try:
        lname = get_langnm(str, lang_detect)

        if lname in SnowballStemmer.languages: # See which languages are supported
            stemmer = SnowballStemmer(lname)  # Choose a language
            str = stemmer.stem(str)  # Stem a word
    except KeyError as e:
        pass
        # print("Unicode error for {0}\n".format(e))

    return str

def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(l, key=alphanum_key)

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    reverse = dict((value, key) for key, value in enums.iteritems())
    enums['reverse_mapping'] = reverse
    return type('Enum', (), enums)


class FEMLFeatures:
    # to_be_removed = "()/.,:!'"  # all characters to be removed
    # file.write('%s: %7d\n' % (word, count))

    # Returned vals: #1: str1 is subset of str2, #2 str2 is subset of str1
    def contains(self, str1, str2, sorted=False):
        str1,_ = normalize_str(str1, sorted)
        str2, _ = normalize_str(str2, sorted)
        return set(str1).issubset(set(str2)), set(str2).issubset(set(str1))

    def contains_freq_term(self, str, freqTerms=None):
        str, _ = normalize_str(str)
        return True if freqTerms != None and str in freqTerms else False

    def contains_specific_freq_term(self, str):
        pass

    def is_matched(self, str):
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

    def hasEncoding_err(self, str1, str2):
        return self.is_matched(str1), self.is_matched(str2)

    def containsAbbr(self, str1, str2):
        abbr1 = re.search(r"\b[A-Z][A-Z\.]{0,}[sr\.]{0,2}\b", str1)
        abbr2 = re.search(r"\b[A-Z][A-Z\.]{0,}[sr\.]{0,2}\b", str2)
        return abbr1, abbr2

    def containsTermsInParenthesis(self, str1, str2):
        tokens1 = re.split('\[|\]|\(|\)', str1)
        tokens2 = re.split('\[|\]|\(|\)', str2)
        return len(tokens1), len(tokens2)

    def containsDashConnected_words(self, str1, str2):
        """
        Hyphenated words are considered to be:
            * a number of word chars
            * followed by any number of: a single hyphen followed by word chars
        """
        dashed1 = re.search(r"\w+(?:-\w+)+", str1)
        dashed2 = re.search(r"\w+(?:-\w+)+", str2)
        return dashed1, dashed2

    def no_of_words(self, str1, str2):
        str1, _ = normalize_str(str1)
        str2, _ = normalize_str(str2)
        return len(set(str1)), len(set(str2))

    def freq_ngram_tokens(self, str1, str2):
        pass

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

    def fagiSim(self, str1, str2):
        pass


class baseMetrics:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, accuracyresults=False):
        self.num_true_predicted_true = [0.0] * len(self.methods)
        self.num_true_predicted_false = [0.0] * len(self.methods)
        self.num_false_predicted_true = [0.0] * len(self.methods)
        self.num_false_predicted_false = [0.0] * len(self.methods)

        self.timers = [0.0] * len(self.methods)
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

        self.algorithms = {
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


    def __del__(self):
        if self.accuracyresults:
            self.file.close()

    @abstractmethod
    def evaluate(self, row, permuted=False, stemming=False, sorting=False, freqTerms=None):
        pass

    @abstractmethod
    def print_stats(self, num_true, num_false):
        pass

    def prediction(self, sim_id, pred_val, real_val):
        result = ""
        var_name = ""
        if real_val == 1.0:
            if pred_val >= self.methods[sim_id - 1][1]:
                var_name = 'num_true_predicted_true'
                result = "\tTRUE"
            else:
                var_name = 'num_true_predicted_false'
                result = "\tFALSE"
        else:
            if pred_val >= self.methods[sim_id - 1][1]:
                var_name = 'num_false_predicted_true'
                result = "\tTRUE"
            else:
                var_name = 'num_false_predicted_false'
                result = "\tFALSE"

        return result, var_name


class calcSotAMetrics(baseMetrics):
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

    def __init__(self, accures):
        super(calcSotAMetrics, self).__init__(accures)

    def generic_evaluator(self, idx, algnm, str1, str2, match):
        start_time = time.time()
        sim = self.algorithms[algnm](str1, str2)
        res, varnm = self.prediction(idx, sim, match)
        self.timers[idx - 1] += (time.time() - start_time)
        self.predictedState[varnm][idx - 1] += 1.0
        return res

    def evaluate(self, row, permuted=False, stemming=False, sorting=False, freqTerms=None):
        tot_res = ""
        real = 1.0 if row['res'] == "TRUE" else 0.0

        # print("{0} - norm: {1}".format(row['s1'], normalize_str(row['s1'])))
        row['s1'] = row['s1'].decode('utf-8')
        row['s2'] = row['s2'].decode('utf-8')
        if stemming:
            row['s1'] = perform_stemming(row['s1'])
            row['s2'] = perform_stemming(row['s2'])
        if sorting:
            a = sorted(row['s1'].split(" "))
            b = sorted(row['s2'].split(" "))
            row['s1'] = " ".join(a)
            row['s2'] = " ".join(b)

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
                file.write("TRUE{0}".format(tot_res + "\n"))
            else:
                file.write("FALSE{0}".format(tot_res + "\n"))

    def print_stats(self, num_true, num_false):
        for idx in range(len(self.methods)):
            try:
                timer = ( self.timers[idx] / float( int( num_true + num_false ) ) ) * 50000.0
                acc = ( self.num_true_predicted_true[idx] + self.num_false_predicted_false[idx] ) / ( num_true + num_false )
                pre = ( self.num_true_predicted_true[idx] ) / ( self.num_true_predicted_true[idx] + self.num_false_predicted_true[idx] )
                rec = ( self.num_true_predicted_true[idx] ) / ( self.num_true_predicted_true[idx] + self.num_true_predicted_false[idx] )
                f1 = 2.0 * ( ( pre * rec ) / ( pre + rec ) )

                print "Metric = Supervised Classifier :" , self.methods[idx][0]
                print "Accuracy =", acc
                print "Precision =", pre
                print "Recall =", rec
                print "F1 =", f1
                print "Processing time per 50K records =", timer
                print ""
                print "| Method\t\t& Accuracy\t& Precision\t& Recall\t& F1-Score\t& Time (50K Pairs)"
                print "||{0}\t& {1}\t& {2}\t& {3}\t& {4}\t& {5}".format(self.methods[idx][0], acc, pre, rec, f1, timer)
                print ""
                sys.stdout.flush()
            except ZeroDivisionError:
                pass
                # print "{0} is divided by zero\n".format(idx)

        # if results:
        #     return self.result


class calcMLCustom(baseMetrics):
    pass


class calcDLearning(baseMetrics):
    pass


class calcSortedMetrics(baseMetrics):
    pass


class Evaluate:
    evaluatorType_action = {
        'SotAMetrics': calcSotAMetrics,
        'MLCustom': calcMLCustom,
        'DLearning': calcDLearning,
        'SortedMetrics': calcSortedMetrics
    }

    def __init__(self, permuted=False, stemming=False, sorting=False, do_printing=False):
        self.permuted = permuted
        self.stemming = stemming
        self.sorting = sorting
        self.only_printing = do_printing

        self.num_true = 0.0
        self.num_false = 0.0
        self.freqTerms = {
            'str1': Counter(), 'str2': Counter(),
            'bi_str1_1': Counter(), 'tri_str1_1': Counter(), 'bi_str1_2': Counter(), 'tri_str1_2': Counter(), 'tri_str1_3': Counter(),
            'bi_str2_1': Counter(), 'tri_str2_1': Counter(), 'bi_str2_2': Counter(), 'tri_str2_2': Counter(), 'tri_str2_3': Counter()
        }
        self.stopwords = []

    def getTMabsPath(self, str):
        return os.path.join(os.path.abspath('../Toponym-Matching'), 'dataset', str)

    def computeInitVals(self, dataset):
        # These are the available languages with stopwords from NLTK
        NLTKlanguages = ["dutch", "finnish", "german", "italian", "portuguese", "spanish", "turkish", "danish",
                         "english", "french", "hungarian", "norwegian", "russian", "swedish"]

        # Just in case more stopword lists are added
        # FREElanguages = []
        # languages = NLTKlanguages + FREElanguages

        for lang in NLTKlanguages:
            self.stopwords.extend(stopwords.words(lang))
        # print(sorted(set(self.stopwords)))

        with open(dataset) as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                    delimiter='\t')
            for row in reader:
                if row['res'] == "TRUE":
                    self.num_true += 1.0
                else:
                    self.num_false += 1.0

                # Calc frequent terms
                # str1
                fterms, stop_words = normalize_str(row['s1'], self.stopwords)
                for term in fterms:
                    self.freqTerms['str1'][term] += 1
                for ngram in list(itertools.chain.from_iterable(
                    [[fterms[i:i + n] for i in range(len(fterms) - (n - 1))] for n in [2, 3]])):
                    if len(ngram) == 2:
                        self.freqTerms['bi_str1_1'][ngram[0]] += 1
                        self.freqTerms['bi_str1_2'][ngram[1]] += 1
                    else:
                        self.freqTerms['tri_str1_1'][ngram[0]] += 1
                        self.freqTerms['tri_str1_2'][ngram[1]] += 1
                        self.freqTerms['tri_str1_3'][ngram[2]] += 1

                # str2
                fterms, stop_words = normalize_str(row['s2'], self.stopwords)
                for term in fterms:
                    self.freqTerms['str2'][term] += 1
                for ngram in list(itertools.chain.from_iterable(
                    [[fterms[i:i + n] for i in range(len(fterms) - (n - 1))] for n in [2, 3]])):
                    if len(ngram) == 2:
                        self.freqTerms['bi_str2_1'][ngram[0]] += 1
                        self.freqTerms['bi_str2_2'][ngram[1]] += 1
                    else:
                        self.freqTerms['tri_str2_1'][ngram[0]] += 1
                        self.freqTerms['tri_str2_2'][ngram[1]] += 1
                        self.freqTerms['tri_str2_3'][ngram[2]] += 1

        if self.only_printing:
            self.do_the_printing()

    def do_the_printing(self):
        print "Printing 10 most common single freq terms..."
        print "str1: {0}".format(self.freqTerms['str1'].most_common(20))
        print "str2: {0}".format(self.freqTerms['str2'].most_common(20))

        print "Printing 10 most common freq terms in bigrams..."
        print "str1 pos 1: {0}".format(self.freqTerms['bi_str1_1'].most_common(20))
        print "\t pos 2: {0}".format(self.freqTerms['bi_str1_2'].most_common(20))
        print "str2 pos 1: {0}".format(self.freqTerms['bi_str2_1'].most_common(20))
        print "\t pos 2: {0}".format(self.freqTerms['bi_str2_2'].most_common(20))

        print "Printing 10 most common freq terms in trigrams..."
        print "str1 pos 1: {0}".format(self.freqTerms['tri_str1_1'].most_common(20))
        print "\t pos 2: {0}".format(self.freqTerms['tri_str1_2'].most_common(20))
        print "\t pos 3: {0}".format(self.freqTerms['tri_str1_3'].most_common(20))
        print "str2 pos 1: {0}".format(self.freqTerms['tri_str2_1'].most_common(20))
        print "\t pos 2: {0}".format(self.freqTerms['tri_str2_2'].most_common(20))
        print "\t pos 3: {0}".format(self.freqTerms['tri_str2_3'].most_common(20))

        with open("freqTerms.csv", "w") as f:
            f.write('str1\t')
            f.write('str2\t')
            f.write('bigram1_pos_1\t')
            f.write('bigram1_pos_2\t')
            f.write('bigram2_pos_1\t')
            f.write('bigram2_pos_2\t')
            f.write('trigram1_pos_1\t')
            f.write('trigram1_pos_2\t')
            f.write('trigram1_pos_3\t')
            f.write('trigram2_pos_1\t')
            f.write('trigram2_pos_2\t')
            f.write('trigram2_pos_3\t')
            f.write('\n')

            sorted_freq_str1_terms = self.freqTerms['str1'].most_common()
            sorted_freq_str2_terms = self.freqTerms['str1'].most_common()
            sorted_freq_bi_str1_1_terms = self.freqTerms['bi_str1_1'].most_common()
            sorted_freq_bi_str1_2_terms = self.freqTerms['bi_str1_2'].most_common()
            sorted_freq_bi_str2_1_terms = self.freqTerms['bi_str2_1'].most_common()
            sorted_freq_bi_str2_2_terms = self.freqTerms['bi_str2_2'].most_common()
            sorted_freq_tri_str1_1_terms = self.freqTerms['tri_str1_1'].most_common()
            sorted_freq_tri_str1_2_terms = self.freqTerms['tri_str1_2'].most_common()
            sorted_freq_tri_str1_3_terms = self.freqTerms['tri_str1_3'].most_common()
            sorted_freq_tri_str2_1_terms = self.freqTerms['tri_str2_1'].most_common()
            sorted_freq_tri_str2_2_terms = self.freqTerms['tri_str2_2'].most_common()
            sorted_freq_tri_str2_3_terms = self.freqTerms['tri_str2_3'].most_common()

            min_top = min(len(sorted_freq_str1_terms),
                        len(sorted_freq_str2_terms),
                        len(sorted_freq_bi_str1_1_terms),
                        len(sorted_freq_bi_str1_2_terms),
                        len(sorted_freq_bi_str2_1_terms),
                        len(sorted_freq_bi_str2_2_terms),
                        len(sorted_freq_tri_str1_1_terms),
                        len(sorted_freq_tri_str1_2_terms),
                        len(sorted_freq_tri_str1_3_terms),
                        len(sorted_freq_tri_str2_1_terms),
                        len(sorted_freq_tri_str2_2_terms),
                        len(sorted_freq_tri_str2_3_terms)
                          )

            for i in range(min_top):
                f.write("{},{}\t".format(sorted_freq_str1_terms[i][0], sorted_freq_str1_terms[i][1]))
                f.write("{},{}\t".format(sorted_freq_str2_terms[i][0], sorted_freq_str2_terms[i][1]))
                f.write("{},{}\t".format(sorted_freq_bi_str1_1_terms[i][0], sorted_freq_bi_str1_1_terms[i][1]))
                f.write("{},{}\t".format(sorted_freq_bi_str1_2_terms[i][0], sorted_freq_bi_str1_2_terms[i][1]))
                f.write("{},{}\t".format(sorted_freq_bi_str2_1_terms[i][0], sorted_freq_bi_str2_1_terms[i][1]))
                f.write("{},{}\t".format(sorted_freq_bi_str2_2_terms[i][0], sorted_freq_bi_str2_2_terms[i][1]))
                f.write("{},{}\t".format(sorted_freq_tri_str1_1_terms[i][0], sorted_freq_tri_str1_1_terms[i][1]))
                f.write("{},{}\t".format(sorted_freq_tri_str1_2_terms[i][0], sorted_freq_tri_str1_2_terms[i][1]))
                f.write("{},{}\t".format(sorted_freq_tri_str1_3_terms[i][0], sorted_freq_tri_str1_3_terms[i][1]))
                f.write("{},{}\t".format(sorted_freq_tri_str2_1_terms[i][0], sorted_freq_tri_str2_1_terms[i][1]))
                f.write("{},{}\t".format(sorted_freq_tri_str2_2_terms[i][0], sorted_freq_tri_str2_2_terms[i][1]))
                f.write("{},{}\t".format(sorted_freq_tri_str2_3_terms[i][0], sorted_freq_tri_str2_3_terms[i][1]))
                f.write('\n')

    def evaluate_metrics(self, dataset='dataset-string-similarity.txt', evalType='SotAMetrics', accuracyresults=False):
        print "Reading dataset..."
        with open(dataset) as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                    delimiter='\t')

            try:
                evalClass = self.evaluatorType_action[evalType](accuracyresults)
            except KeyError:
                print("Unkown method")
                return

            for row in reader:
                evalClass.evaluate(row, self.permuted, self.stemming, self.sorting, self.freqTerms)
            evalClass.print_stats(self.num_true, self.num_false)


def main(args):
    dataset_path = args['-d']

    eval = Evaluate(args['--permuted'], args['--stemming'], args['--sort'], args['--print'])
    full_dataset_path = eval.getTMabsPath(dataset_path)

    if os.path.isfile(full_dataset_path):
        eval.computeInitVals(full_dataset_path)
        if args['--print']:
            sys.exit()

        eval.evaluate_metrics(full_dataset_path, args['--ev'], args['--accuracyresults'])

        # Supervised machine learning
        if args['--ev'] == "SotAMetrics":
            for method in ['rf', 'et', 'svm', 'xgboost']:
                evaluate_classifier(dataset=full_dataset_path, method=method, accuracyresults=args['--accuracyresults'],
                                    permuted=args['--permuted'], results=False)
    else:
        print "No file {0} exists!!!\n".format(full_dataset_path)


if __name__ == "__main__":
    arguments = docopt(__doc__, version='FE-ML 0.1')
    main(arguments)
