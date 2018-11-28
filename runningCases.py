import os, sys
import csv
from collections import Counter
import itertools
import json
from operator import is_not
from functools import partial

from nltk.corpus import stopwords

import femlAlgorithms as femlAlgs
from staticArguments import perform_stemming, normalize_str, sorted_nicely


class Evaluator:
    evaluatorType_action = {
        'SotAMetrics': femlAlgs.calcSotAMetrics,
        'SotAML': femlAlgs.calcSotAML,
        'customFEML': femlAlgs.calcCustomFEML,
        'DLearning': femlAlgs.calcDLearning,
    }

    def __init__(self, permuted=False, stemming=False, sorting=False, do_printing=False):
        self.permuted = permuted
        self.stemming = stemming
        self.sorting = sorting
        self.only_printing = do_printing

        self.freqTerms = {
            'gram': Counter(),
            '2gram_1': Counter(), '3gram_1': Counter(),
            '2gram_2': Counter(), '3gram_2': Counter(), '3gram_3': Counter(),
        }
        self.stop_words = []
        self.abbr = {'A': [], 'B': []}
        self.fsorted = None
        self.evalClass = None

    @staticmethod
    def getTMabsPath(ds):
        return os.path.join(os.path.abspath('../Toponym-Matching'), 'dataset', ds)

    @staticmethod
    def getRelativeDirpathToWorking(ds):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), ds)

    def initialize(self, dataset, evalType='SotAMetrics', njobs=2, accuracyresults=False):
        try:
            self.evalClass = self.evaluatorType_action[evalType](njobs, accuracyresults)
        except KeyError:
            print("Unkown method")
            return 1

        # These are the available languages with stopwords from NLTK
        NLTKlanguages = ["dutch", "finnish", "german", "italian", "portuguese", "spanish", "turkish", "danish",
                         "english", "french", "hungarian", "norwegian", "russian", "swedish"]
        for lang in NLTKlanguages:
            self.stop_words.extend(stopwords.words(lang))

        FREElanguages = [
            'zh', 'ja', 'id', 'fa', 'ar', 'bn', 'ro', 'th', 'el', 'hi', 'gl', 'hy', 'ko', 'yo', 'vi',
            'sw', 'so', 'he', 'ha', 'br', 'af', 'ku', 'ms', 'tl', 'ur'
        ]
        if os.path.isfile(os.path.join(os.getcwd(), 'stopwords-iso.json')):
            with open("stopwords-iso.json", "r") as read_file:
                data = json.load(read_file)
                for lang in FREElanguages:
                    self.stop_words.extend(data[lang])

        if self.only_printing:
            self.fsorted = open('sorted.csv', 'w')
            self.fsorted.write("Original_A\tSorted_A\tOriginal_B\tSorted_B\n")

        feml = femlAlgs.FEMLFeatures()
        with open(dataset) as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                    delimiter='\t')
            for row in reader:
                self.evalClass.preprocessing(row)

                # Calc frequent terms
                # (str1, str2)
                for str in ['s1', 's2']:
                    fterms, stop_words = normalize_str(row[str], self.stop_words)
                    for term in fterms:
                        self.freqTerms['gram'][term] += 1
                    for gram in list(itertools.chain.from_iterable(
                        [[fterms[i:i + n] for i in range(len(fterms) - (n - 1))] for n in [2, 3]])):
                        if len(gram) == 2:
                            self.freqTerms['2gram_1'][gram[0]] += 1
                            self.freqTerms['2gram_2'][gram[1]] += 1
                        else:
                            self.freqTerms['3gram_1'][gram[0]] += 1
                            self.freqTerms['3gram_2'][gram[1]] += 1
                            self.freqTerms['3gram_3'][gram[2]] += 1

                # calc the number of abbr that exist
                self.abbr['A'].append(feml.containsAbbr(row['s1']))
                self.abbr['B'].append(feml.containsAbbr(row['s2']))

                if self.only_printing:
                    self.fsorted.write(row['s1'])
                    self.fsorted.write("\t")
                    self.fsorted.write(" ".join(sorted_nicely(row['s1'].split(" "))))
                    self.fsorted.write("\t")
                    self.fsorted.write(row['s2'])
                    self.fsorted.write("\t")
                    self.fsorted.write(" ".join(sorted_nicely(row['s2'].split(" "))))
                    self.fsorted.write("\n")

        if self.only_printing:
            self.fsorted.close()

        return 0

    def do_the_printing(self):
        print "Printing 10 most common single freq terms..."
        print "gram: {0}".format(self.freqTerms['gram'].most_common(20))

        print "Printing 10 most common freq terms in bigrams..."
        print "bi-gram pos 1: {0}".format(self.freqTerms['2gram_1'].most_common(20))
        print "\t pos 2: {0}".format(self.freqTerms['2gram_2'].most_common(20))

        print "Printing 10 most common freq terms in trigrams..."
        print "tri-gram pos 1: {0}".format(self.freqTerms['3gram_1'].most_common(20))
        print "\t pos 2: {0}".format(self.freqTerms['3gram_2'].most_common(20))
        print "\t pos 3: {0}".format(self.freqTerms['3gram_3'].most_common(20))

        print "Number of abbr found: {0}".format(len(filter(partial(is_not, '-'), self.abbr['A'])) +
                                                 len(filter(partial(is_not, '-'), self.abbr['B'])))

        with open("freqTerms.csv", "w") as f:
            f.write('gram\t')
            f.write('bigram_pos_1\t')
            f.write('bigram_pos_2\t')
            f.write('trigram_pos_1\t')
            f.write('trigram_pos_2\t')
            f.write('trigram_pos_3')
            f.write('\n')

            sorted_freq_gram_terms = self.freqTerms['gram'].most_common()
            sorted_freq_bigram_terms_pos1 = self.freqTerms['2gram_1'].most_common()
            sorted_freq_bigram_terms_pos2 = self.freqTerms['2gram_2'].most_common()
            sorted_freq_trigram_terms_pos1 = self.freqTerms['3gram_1'].most_common()
            sorted_freq_trigram_terms_pos2 = self.freqTerms['3gram_2'].most_common()
            sorted_freq_trigram_terms_pos3 = self.freqTerms['3gram_3'].most_common()

            min_top = min(
                len(sorted_freq_gram_terms),
                len(sorted_freq_bigram_terms_pos1),
                len(sorted_freq_bigram_terms_pos2),
                len(sorted_freq_trigram_terms_pos1),
                len(sorted_freq_trigram_terms_pos2),
                len(sorted_freq_trigram_terms_pos3),
            )

            for i in range(min_top):
                f.write("{},{}\t".format(sorted_freq_gram_terms[i][0], sorted_freq_gram_terms[i][1]))
                f.write("{},{}\t".format(sorted_freq_bigram_terms_pos1[i][0], sorted_freq_bigram_terms_pos1[i][1]))
                f.write("{},{}\t".format(sorted_freq_bigram_terms_pos2[i][0], sorted_freq_bigram_terms_pos2[i][1]))
                f.write("{},{}\t".format(sorted_freq_trigram_terms_pos1[i][0], sorted_freq_trigram_terms_pos1[i][1]))
                f.write("{},{}\t".format(sorted_freq_trigram_terms_pos2[i][0], sorted_freq_trigram_terms_pos2[i][1]))
                f.write("{},{}\t".format(sorted_freq_trigram_terms_pos3[i][0], sorted_freq_trigram_terms_pos3[i][1]))
                f.write('\n')

        with open("abbr.csv", "w") as f:
            f.write('strA\tstrB\tline_pos\n')
            for i in range(min(len(self.abbr['A']), len(self.abbr['B']))):
                if self.abbr['A'][i] != '-' or self.abbr['B'][i] != '-':
                    f.write("{}\t{}\t{}\n".format(self.abbr['A'][i], self.abbr['B'][i], i))

    def evaluate_metrics(self, dataset='dataset-string-similarity.txt'):
        if self.evalClass is not None:
            print "Reading dataset..."
            relpath = self.getRelativeDirpathToWorking(dataset)
            with open(relpath) as csvfile:
                reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                        delimiter='\t')

                for row in reader:
                    self.evalClass.evaluate(row, self.sorting, self.stemming, self.permuted, self.freqTerms)
                if hasattr(self.evalClass, "train_classifiers"): self.evalClass.train_classifiers()
                self.evalClass.print_stats()

