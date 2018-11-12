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
  --sorted                  Sort alphanumerically.
  --ev <evaluator_type>     Type of experiments to conduct. [default: SotAMetrics]

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

# import configparser
import helpers
from docopt import docopt
import os, sys
import csv
import time
from collections import Counter
from nltk import SnowballStemmer, wordpunct_tokenize
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import stopwords
from langdetect import detect, lang_detect_exception
import pycountry
import re
from abc import ABCMeta, abstractmethod

from featureclassifiers import evaluate_classifier
from datasetcreator import damerau_levenshtein, jaccard, jaro, jaro_winkler,monge_elkan, cosine, strike_a_match, \
    soft_jaccard, sorted_winkler, permuted_winkler, skipgram, davies
from datasetcreator import detect_alphabet, fields


def get_langnm(str, lang_detect=False):
    lname = 'english'
    try:
        lname = pycountry.languages.get(alpha_2=detect(str)).name.lower() if lang_detect else 'english'
    except lang_detect_exception.LangDetectException as e:
        print(e)

    return lname

# Clean the string from stopwords, puctuations based on language detections feature
# Returned values #1: non-stopped words, #2: stopped words
def normalize_str(str, sorted=False, lang_detect=False):
    languagenm = get_langnm(str) if lang_detect else 'english'

    tokens = wordpunct_tokenize(str)
    words = [word.lower() for word in tokens if word.isalpha()]
    stopwords_set = set(stopwords.words(languagenm))

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
    # if not os.path.exists("statistics"):
    #     os.makedirs("statistics")
    # to_be_removed = "()/.,:!'"  # all characters to be removed
    # file.write('%s: %7d\n' % (word, count))

    # giann: extracting 2,3,4-grams
    def extract_ngrams(input='dataset/allCountries.txt', output='dataset_ngrams', remove_punct=1, numberList=[2]):

        """Extracts dataset statistics.
        numberList is a list of integers k prescribing the k-grams to be extracted
        Only 2,3,4 values are permitted
        Currently works for only English and includes alternate names
        """

        if not os.path.exists("statistics"):
            os.makedirs("statistics")

        file = {}
        ngramCounter = Counter()

        if len(numberList) > 3:
            print (
                "Error: Too many different n-gram types to calculate!\n Currently, we can only calculate 2,3,4-grams!")
            return 1
        for ngr in numberList:
            if (ngr < 2) or (ngr > 4):
                print (
                    "Error: Too many different n-gram types to calculate!\n Currently, we can only calculate 2,3,4-grams!")
                return 1
            else:
                if (remove_punct == 1):
                    file[ngr] = open("statistics/" + output + str(ngr) + "_rempunct.txt", "w+")
                else:
                    file[ngr] = open("statistics/" + output + str(ngr) + ".txt", "w+")
                ngramCounter[ngr] = Counter()

        cou = 0;
        with open(input) as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=fields, delimiter='\t')
            to_be_removed = "()/.,:!'"  # all characters to be removed
            for row in reader:
                name, altname = row['asciiname'], row['alternatenames']

                if (remove_punct == 1):
                    for c in to_be_removed:
                        name = name.replace(c, ' ')
                        altname = altname.replace(c, ' ')

                tokens1 = wordpunct_tokenize(name)
                tokens2 = wordpunct_tokenize(name)

                tokens1 = tokens1 + tokens2

                # print (tokens1)

                finder1 = BigramCollocationFinder.from_words(tokens1)
                # finder2 = TrigramCollocationFinder.from_words(tokens1)
                # finder3 = QuadgramCollocationFinder.from_words(tokens1)

                # finder1.apply_freq_filter(50)
                # finder2.apply_freq_filter(50)
                # finder3.apply_freq_filter(50)

                # print (finder1.ngram_fd)

                ngramCounter[2] = ngramCounter[2] + finder1.ngram_fd
                # ngramCounter[3] = ngramCounter[3] + finder2.ngram_fd
                # ngramCounter[4] = ngramCounter[4] + finder3.ngram_fd

                # for key, value in ngramCounter[2].items():
                #    print(key, value)

                # cou += 1
                # if (cou == 5):
                #    break

        for ngrm in numberList:
            for ngram, count in ngramCounter[ngrm].most_common(1000):
                file[ngrm].write('%s: %7d\n' % (ngram, count))

        return 0


    # Returned vals: #1: str1 is subset of str2, #2 str2 is subset of str1
    def contains(self, str1, str2):
        str1,_ = normalize_str(str1)
        str2, _ = normalize_str(str2)

        return set(str1).issubset(set(str2)), set(str2).issubset(set(str1))

    def contains_freq_term(self, str1, str2):
        term_counter = Counter()
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

    def __del__(self):
        if self.accuracyresults:
            self.file.close()

    @abstractmethod
    def evaluate(self, row, permuted=False, stemming=False, sorted=False):
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

    def __init__(self):
        super(calcSotAMetrics, self).__init__()

    def evaluate(self, row, permuted=False, stemming=False, sorted=False):
        tot_res = ""
        real = 1.0 if row['res'] == "TRUE" else 0.0

        # print("{0} - norm: {1}".format(row['s1'], normalize_str(row['s1'])))
        row['s1'] = row['s1'].decode('utf-8')
        row['s2'] = row['s2'].decode('utf-8')
        if stemming:
            row['s1'] = perform_stemming(row['s1'])
            row['s2'] = perform_stemming(row['s2'])

        start_time = time.time()
        sim1 = damerau_levenshtein(row['s1'], row['s2'])
        res, varnm = self.prediction(1, sim1, real)
        self.timers[1 - 1] += (time.time() - start_time)
        self.predictedState[varnm][0] += 1.0
        tot_res += res

        start_time = time.time()
        sim8 = jaccard(row['s1'], row['s2'])
        res, varnm = self.prediction(8, sim8, real)
        self.timers[8 - 1] += (time.time() - start_time)
        self.predictedState[varnm][8 - 1] += 1.0
        tot_res += res

        start_time = time.time()
        sim2 = jaro(row['s1'], row['s2'])
        res, varnm = self.prediction(2, sim2, real)
        self.timers[2 - 1] += (time.time() - start_time)
        self.predictedState[varnm][2 - 1] += 1.0
        tot_res += res

        start_time = time.time()
        sim3 = jaro_winkler(row['s1'], row['s2'])
        res, varnm = self.prediction(3, sim3, real)
        self.timers[3 - 1] += (time.time() - start_time)
        self.predictedState[varnm][3 - 1] += 1.0
        tot_res += res

        start_time = time.time()
        sim4 = jaro_winkler(row['s1'][::-1], row['s2'][::-1])
        res, varnm = self.prediction(4, sim4, real)
        self.timers[4 - 1] += (time.time() - start_time)
        self.predictedState[varnm][4 - 1] += 1.0
        tot_res += res

        start_time = time.time()
        sim11 = monge_elkan(row['s1'], row['s2'])
        res, varnm = self.prediction(11, sim11, real)
        self.timers[11 - 1] += (time.time() - start_time)
        self.predictedState[varnm][11 - 1] += 1.0
        tot_res += res

        start_time = time.time()
        sim7 = cosine(row['s1'], row['s2'])
        res, varnm = self.prediction(7, sim7, real)
        self.timers[7 - 1] += (time.time() - start_time)
        self.predictedState[varnm][7 - 1] += 1.0
        tot_res += res

        start_time = time.time()
        sim9 = strike_a_match(row['s1'], row['s2'])
        res, varnm = self.prediction(9, sim9, real)
        self.timers[9 - 1] += (time.time() - start_time)
        self.predictedState[varnm][9 - 1] += 1.0
        tot_res += res

        start_time = time.time()
        sim12 = soft_jaccard(row['s1'], row['s2'])
        res, varnm = self.prediction(12, sim12, real)
        self.timers[12 - 1] += (time.time() - start_time)
        self.predictedState[varnm][12 - 1] += 1.0
        tot_res += res

        start_time = time.time()
        sim5 = sorted_winkler(row['s1'], row['s2'])
        res, varnm = self.prediction(5, sim5, real)
        self.timers[5 - 1] += (time.time() - start_time)
        self.predictedState[varnm][5 - 1] += 1.0
        tot_res += res

        if permuted:
            start_time = time.time()
            sim6 = permuted_winkler(row['s1'], row['s2'])
            res, varnm = self.prediction(6, sim6, real)
            self.timers[6 - 1] += (time.time() - start_time)
            self.predictedState[varnm][6 - 1] += 1.0
            tot_res += res

        start_time = time.time()
        sim10 = skipgram(row['s1'], row['s2'])
        res, varnm = self.prediction(10, sim10, real)
        self.timers[10 - 1] += (time.time() - start_time)
        self.predictedState[varnm][10 - 1] += 1.0
        tot_res += res

        start_time = time.time()
        sim13 = davies(row['s1'], row['s2'])
        res, varnm = self.prediction(13, sim13, real)
        self.timers[13 - 1] += (time.time() - start_time)
        self.predictedState[varnm][13 - 1] += 1.0
        tot_res += res

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
                print "{0} is divided by zero\n".format(idx)

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

    def __init__(self, permuted=False, stemming=False, sorted=False):
        self.permuted = permuted
        self.stemming = stemming
        self.sorted = sorted

        self.num_true = 0.0
        self.num_false = 0.0

    def getTMabsPath(self, str):
        return os.path.join(os.path.abspath('../Toponym-Matching'), 'dataset', str)

    def computeTrueFalseVals(self, dataset):
        with open(dataset) as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                    delimiter='\t')
            for row in reader:
                if row['res'] == "TRUE":
                    self.num_true += 1.0
                else:
                    self.num_false += 1.0

    def evaluate_metrics(self, dataset='dataset-string-similarity.txt', evalType='SotAMetrics', accuracyresults=False):
        self.computeTrueFalseVals(dataset)

        print "Reading dataset..."
        with open(dataset) as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                    delimiter='\t')

            try:
                evalClass = self.evaluatorType_action[evalType]()
            except KeyError:
                print("Unkown method")
                return

            for row in reader:
                evalClass.evaluate(row, self.permuted, self.stemming, self.sorted)
            evalClass.print_stats(self.num_true, self.num_false)


def main(args):
    dataset_path = args['-d']

    eval = Evaluate(args['--permuted'], args['--stemming'], args['--sorted'])
    full_dataset_path = eval.getTMabsPath(dataset_path)

    if os.path.isfile(full_dataset_path):
        eval.evaluate_metrics(full_dataset_path, args['--ev'])

        # Supervised machine learning
        if args['--ev'] == "SotAMetrics":
            for method in ['rf', 'et', 'svm', 'xgboost']:
                evaluate_classifier(dataset=full_dataset_path, method=method, accuracyresults=True, results=False)
    else:
        print "No file {0} exists!!!\n".format(full_dataset_path)


if __name__ == "__main__":
    arguments = docopt(__doc__, version='FE-ML 0.1')
    main(arguments)
