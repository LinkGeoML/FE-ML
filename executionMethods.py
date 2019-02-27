# -*- coding: utf-8 -*-

from __future__ import print_function
import os, sys
from collections import Counter
import json
from operator import is_not
from functools import partial
# import unicodedata
import pandas as pd

from nltk.corpus import stopwords

from femlAlgorithms import *
from helpers import perform_stemming, normalize_str, sorted_nicely, getRelativePathtoWorking, getTMabsPath
from external.datasetcreator import detect_alphabet, strip_accents, filter_dataset, build_dataset_from_geonames
from helpers import StaticValues


class Evaluator:
    evaluatorType_action = {
        'SotAMetrics': calcSotAMetrics,
        'SotAML': calcSotAML,
        'customFEML': calcCustomFEML,
        'DLearning': calcDLearning,
        'TestMetrics': testMetrics,
        'customFEMLExtended': calcCustomFEMLExtended,
    }

    def __init__(self, ml_algs, sorting=False, stemming=False, canonical=False, permuted=False, do_printing=False,
                 only_latin=False):
        self.ml_algs = [x for x in ml_algs.split(',')]
        self.permuted = permuted
        self.stemming = stemming
        self.canonical = canonical
        self.sorting = sorting
        self.only_printing = do_printing
        self.latin = only_latin

        self.termfrequencies = {
            'gram': Counter(),
            '2gram_1': Counter(), '3gram_1': Counter(),
            '2gram_2': Counter(), '3gram_2': Counter(), '3gram_3': Counter(),
        }
        self.termsperalphabet = {}
        self.stop_words = []
        self.abbr = {'A': [], 'B': []}
        self.evalClass = None

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

        with open(dataset) as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                    delimiter='\t')
            for row in reader:
                self.evalClass.preprocessing(row)

    def do_the_printing(self):
        if not os.path.exists("output"):
            os.makedirs("output")

        print("Printing 10 most common single freq terms...")
        print( "gram: {0}".format(self.termfrequencies['gram'].most_common(20)))

        print("Printing 10 most common freq terms in bigrams...")
        print("bi-gram pos 1: {0}".format(self.termfrequencies['2gram_1'].most_common(20)))
        print("\t pos 2: {0}".format(self.termfrequencies['2gram_2'].most_common(20)))

        print("Printing 10 most common freq terms in trigrams...")
        print("tri-gram pos 1: {0}".format(self.termfrequencies['3gram_1'].most_common(20)))
        print("\t pos 2: {0}".format(self.termfrequencies['3gram_2'].most_common(20)))
        print("\t pos 3: {0}".format(self.termfrequencies['3gram_3'].most_common(20)))

        print("Number of abbr found: {0}".format(len(filter(partial(is_not, '-'), self.abbr['A'])) +
                                                 len(filter(partial(is_not, '-'), self.abbr['B']))))

        with open("./output/freqTerms.csv", "w") as f:
            f.write('gram\t')
            f.write('bigram_pos_1\t')
            f.write('bigram_pos_2\t')
            f.write('trigram_pos_1\t')
            f.write('trigram_pos_2\t')
            f.write('trigram_pos_3')
            f.write('\n')

            sorted_freq_gram_terms = self.termfrequencies['gram'].most_common()
            sorted_freq_bigram_terms_pos1 = self.termfrequencies['2gram_1'].most_common()
            sorted_freq_bigram_terms_pos2 = self.termfrequencies['2gram_2'].most_common()
            sorted_freq_trigram_terms_pos1 = self.termfrequencies['3gram_1'].most_common()
            sorted_freq_trigram_terms_pos2 = self.termfrequencies['3gram_2'].most_common()
            sorted_freq_trigram_terms_pos3 = self.termfrequencies['3gram_3'].most_common()

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

        with open("./output/abbr.csv", "w") as f:
            f.write('strA\tstrB\tline_pos\n')
            for i in range(min(len(self.abbr['A']), len(self.abbr['B']))):
                if self.abbr['A'][i] != '-' or self.abbr['B'][i] != '-':
                    f.write("{}\t{}\t{}\n".format(self.abbr['A'][i], self.abbr['B'][i], i))

    def evaluate_metrics(self, dataset='dataset-string-similarity.txt'):
        if self.evalClass is not None:
            print( "Reading dataset...")
            relpath = getRelativePathtoWorking(dataset)
            self.evalClass.freq_terms_list()

            with open(relpath) as csvfile:
                reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                        delimiter='\t')

                thres = 'orig'
                if self.sorting:
                    thres = 'sorted'

                for row in reader:
                    self.evalClass.evaluate(
                        row, self.sorting, self.stemming, self.canonical, self.permuted, self.termfrequencies, thres
                    )
                if hasattr(self.evalClass, "train_classifiers"): self.evalClass.train_classifiers(self.ml_algs, polynomial=False, standardize=True)
                self.evalClass.print_stats()

    def evaluate_metrics_with_various_thres(self, dataset='dataset-string-similarity.txt'):
        if self.evalClass is not None:
            print("Reading dataset...")
            relpath = getRelativePathtoWorking(dataset)

            start_time = time.time()
            with open(relpath) as csvfile:
                reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                        delimiter='\t')

                all_res = {}
                for m in StaticValues.methods: all_res[m[0]] = []
                for i in range(5, 99, 5):
                    print('Computing stats for threshold {0}...'.format(float(i / 100.0)))

                    csvfile.seek(0)
                    for row in reader:
                        self.evalClass.evaluate(
                            row, self.sorting, self.stemming, self.canonical, self.permuted, self.termfrequencies, float(i / 100.0)
                        )
                    if hasattr(self.evalClass, "train_classifiers"): self.evalClass.train_classifiers(self.ml_algs)
                    tmp_res = self.evalClass.get_stats()

                    for key, val in tmp_res.items():
                        all_res[key].append([float(i / 100.0), val])

                    self.evalClass.reset_vars()

            print('The process took {0:.2f} sec'.format(time.time() - start_time))
            for k, val in all_res.items():
                if len(val) == 0:
                    print('{0} is empty'.format(k))
                    continue

                print(k, max(val, key=lambda x: x[1][0]))

    def evaluate_sorting_with_various_thres(self, dataset='dataset-string-similarity.txt'):
        if self.evalClass is not None:
            print("Reading dataset...")
            relpath = getRelativePathtoWorking(dataset)

            start_time = time.time()
            with open(relpath) as csvfile:
                reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                        delimiter='\t')

                all_res = {}
                for m in StaticValues.methods: all_res[m[0]] = []
                for i in range(55, 86, 5):
                    print('Computing stats for threshold {0}...'.format(float(i / 100.0)))

                    csvfile.seek(0)
                    for row in reader:
                        if self.latin and (row['a1'] != 'LATIN' or row['a2'] != 'LATIN'): continue

                        self.evalClass.evaluate_sorting(row, float(i / 100.0), self.stemming, self.permuted)
                    if hasattr(self.evalClass, "train_classifiers"): self.evalClass.train_classifiers(self.ml_algs)
                    tmp_res = self.evalClass.get_stats()

                    for key, val in tmp_res.items():
                        all_res[key].append([float(i / 100.0), val])

                    self.evalClass.reset_vars()

            print('The process took {0:.2f} sec'.format(time.time() - start_time))
            for k, val in all_res.items():
                if len(val) == 0:
                    print('{0} is empty'.format(k))
                    continue

                print(k, max(val, key=lambda x: x[1][0]))

    def test_cases(self, dataset, test_case):
        if test_case - 1 == 0:
            if not os.path.exists("output"):
                os.makedirs("output")

            abbr_stats = Counter()
            fscoresless = open("./output/lower_score_on_transformation.csv", "w")
            fscoresless.write("strA\tstrB\tsorted_strA\tsorted_strB\n")

            feml = FEMLFeatures()
            with open(dataset) as csvfile:
                reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                        delimiter='\t')

                for row in reader:
                    if feml.cmp_score_after_transformation(row, sorting=True, canonical=True) is False:
                        a = strip_accents(row['s1'].decode('utf8'))
                        b = strip_accents(row['s2'].decode('utf8'))

                        fscoresless.write("{}\t{}\t{}\t{}\n".format(
                            row['s1'], row['s2'],
                            " ".join(sorted_nicely(a.lower().split(" "))).encode('ASCII', 'ignore'),
                            " ".join(sorted_nicely(b.lower().split(" "))).encode('ASCII', 'ignore')
                        ))

                    # Calc frequent terms
                    # (str1, str2)
                    for sstr in ['s1', 's2']:
                        ab = detect_alphabet(row[sstr])
                        if ab not in self.termsperalphabet.keys():
                            self.termsperalphabet[ab] = {
                                'gram': Counter(),
                                '2gram_1': Counter(), '3gram_1': Counter(),
                                '2gram_2': Counter(), '3gram_2': Counter(), '3gram_3': Counter(),
                            }

                        fterms, stop_words = normalize_str(row[sstr], self.stop_words)
                        for term in fterms:
                            self.termsperalphabet[ab]['gram'][term] += 1
                        for gram in list(itertools.chain.from_iterable(
                                [[fterms[i:i + n] for i in range(len(fterms) - (n - 1))] for n in [2, 3]])):
                            if len(gram) == 2:
                                self.termsperalphabet[ab]['2gram_1'][gram[0]] += 1
                                self.termsperalphabet[ab]['2gram_2'][gram[1]] += 1
                            else:
                                self.termsperalphabet[ab]['3gram_1'][gram[0]] += 1
                                self.termsperalphabet[ab]['3gram_2'][gram[1]] += 1
                                self.termsperalphabet[ab]['3gram_3'][gram[2]] += 1

                    # calc the number of abbr that exist
                    freq1 = feml.containsAbbr(row['s1'])
                    freq2 = feml.containsAbbr(row['s2'])
                    if freq1 != '-': abbr_stats[freq1] += 1
                    if freq2 != '-' and freq1 != freq2: abbr_stats[freq2] += 1

            if not fscoresless.closed:
                fscoresless.close()

            with open("./output/abbr.csv", "w") as f:
                f.write('abbr\tcount\n')
                for value, count in abbr_stats.most_common():
                    f.write("{}\t{}\n".format(value, count))

            for k, v in self.termsperalphabet.items():
                with open("./output/freqterms_for_{}.csv".format(k), "w") as f:
                    f.write('gram\t')
                    f.write('bigram_pos_1\t')
                    f.write('bigram_pos_2\t')
                    f.write('trigram_pos_1\t')
                    f.write('trigram_pos_2\t')
                    f.write('trigram_pos_3')
                    f.write('\n')

                    sorted_freq_gram_terms = v['gram'].most_common()
                    sorted_freq_bigram_terms_pos1 = v['2gram_1'].most_common()
                    sorted_freq_bigram_terms_pos2 = v['2gram_2'].most_common()
                    sorted_freq_trigram_terms_pos1 = v['3gram_1'].most_common()
                    sorted_freq_trigram_terms_pos2 = v['3gram_2'].most_common()
                    sorted_freq_trigram_terms_pos3 = v['3gram_3'].most_common()

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
                        f.write(
                            "{},{}\t".format(sorted_freq_trigram_terms_pos1[i][0], sorted_freq_trigram_terms_pos1[i][1]))
                        f.write(
                            "{},{}\t".format(sorted_freq_trigram_terms_pos2[i][0], sorted_freq_trigram_terms_pos2[i][1]))
                        f.write(
                            "{},{}\t".format(sorted_freq_trigram_terms_pos3[i][0], sorted_freq_trigram_terms_pos3[i][1]))
                        f.write('\n')
        elif test_case - 1 == 1:
            ngram_stats = {
                '2gram': Counter(), '3gram': Counter(), '4gram': Counter(),
                'gram_token': Counter(), '2gram_token': Counter(), '3gram_token': Counter()
            }
            abbr_stats = Counter()
            orig_strs = {}
            no_dashed_strs = 0

            with open(dataset) as csvfile:
                reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                        delimiter='\t')

                feml = FEMLFeatures()
                for row in reader:
                    # row['s1'], row['s2'] = femlAlgs.transform(row['s1'], row['s2'], canonical=True)

                    for sstr in ['s1', 's2']:
                        # calc the number of abbr that exist
                        abbr_str = feml.containsAbbr(row[sstr])
                        if abbr_str != '-':
                            if abbr_str not in orig_strs.keys(): orig_strs[abbr_str] = []
                            abbr_stats[abbr_str] += 1
                            orig_strs[abbr_str].append(row[sstr])

                        # search for dashes in strings
                        no_dashed_strs += feml.containsDashConnected_words(row[sstr])

                        row[sstr] = transform_str(row[sstr], canonical=True)
                        ngram_tokens, _ = normalize_str(row[sstr], self.stop_words)

                        for term in ngram_tokens:
                            ngram_stats['gram_token'][term] += 1
                        for gram in list(itertools.chain.from_iterable(
                                [[ngram_tokens[i:i + n] for i in range(len(ngram_tokens) - (n - 1))]
                                 for n in [2, 3]])
                        ):
                            if len(gram) == 2:
                                ngram_stats['2gram_token'][' '.join(gram)] += 1
                            else:
                                ngram_stats['3gram_token'][' '.join(gram)] += 1

                        # ngrams chars
                        # ngrams = zip(*[''.join(strA_ngrams_tokens)[i:] for i in range(n) for n in [2, 3, 4]])
                        for gram in list(itertools.chain.from_iterable(
                                [[''.join(ngram_tokens)[i:i + n] for i in range(len(''.join(ngram_tokens)) - (n - 1))]
                                 for n in [2, 3, 4]])
                        ):
                            if len(gram) == 2:
                                ngram_stats['2gram'][gram] += 1
                            elif len(gram) == 3:
                                ngram_stats['3gram'][gram] += 1
                            elif len(gram) == 4:
                                ngram_stats['4gram'][gram] += 1

            print("Found {} dashed words in the dataset.".format(no_dashed_strs))

            with open("./output/abbr.csv", "w+") as f:
                f.write('abbr\tcount\tstr\n')
                for value, count in abbr_stats.most_common():
                    f.write("{0}\t{1}\t{2}\n".format(value, count, ','.join(orig_strs[value][:10])))

            for nm in ngram_stats.keys():
                with open("./output/{0}s.csv".format(nm), "w+") as f:
                    f.write('gram\tcount\n')
                    for value, count in ngram_stats[nm].most_common():
                        f.write("{}\t{}\n".format(value.encode('utf8'), count))
        elif test_case - 1 == 2:
            if self.evalClass is not None:
                print("Reading dataset...")
                relpath = getRelativePathtoWorking(dataset)
                self.evalClass.freq_terms_list()

                with open(relpath) as csvfile:
                    reader = csv.DictReader(csvfile,
                                            fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                            delimiter='\t')

                    all_res = {}
                    for m in StaticValues.methods: all_res[m[0]] = []
                    feml = FEMLFeatures()
                    for n in [3.34] + list(range(4, 8)):
                        combs = [tuple(float(x/10.0) for x in seq) for seq in itertools.product([1, 2, 3, 4, 5, 2.5, 3.33], repeat=2) if sum(seq) == (10 - n)]

                        for w in combs:
                            w = (float(n/10.0), ) + w
                            feml.update_weights(w)
                            print('Computing stats for weights ({})'.format(','.join(map(str, w))))
                            print('Computing stats for threshold', end='')

                            separator = ''
                            start_time = time.time()
                            for i in range(30, 91, 5):
                                print('{0} {1}'.format(separator, float(i / 100.0)), end='')
                                separator = ','
                                #  required for python before 3.3
                                sys.stdout.flush()

                                csvfile.seek(0)
                                for row in reader:
                                    self.evalClass.evaluate(
                                        row, self.sorting, self.stemming, self.canonical, self.permuted, self.termfrequencies,
                                        float(i / 100.0)
                                    )
                                if hasattr(self.evalClass, "train_classifiers"): self.evalClass.train_classifiers(self.ml_algs)
                                tmp_res = self.evalClass.get_stats()

                                for key, val in tmp_res.items():
                                    all_res[key].append([float(i / 100.0), val])

                                self.evalClass.reset_vars()

                            print('\nThe process for weight ({0}) took {1:.2f} sec'.format(','.join(map(str, w)), time.time() - start_time))
                            for k, val in all_res.items():
                                if len(val) == 0:
                                    continue

                                print(k, max(val, key=lambda x: x[1][0]))

                            for m in StaticValues.methods: all_res[m[0]][:] = []
        else:
            print("Test #{} does not exist!!! Please choose a valid test to execute.".format(test_case))

    def print_false_posneg(self, datasets):
        if not os.path.exists("output"):
            os.makedirs("output")

        if len(datasets) == 2:
            reader = pd.read_csv(getTMabsPath(datasets[0]), sep='\t', names=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"])
            results = pd.read_csv(datasets[1], sep='\t', names=["res1", "res2"])

            print("No of rows for dataset: {0}".format(results.shape[0]/2))
            resultDf = pd.concat([results.iloc[results.shape[0]/2:], results.iloc[:results.shape[0]/2]], ignore_index=True)
            mismatches = pd.concat([reader, resultDf], axis=1)

            negDf = mismatches[
                (not self.latin or mismatches.a1 == 'LATIN') & (not self.latin or mismatches.a2 == 'LATIN') &
                (mismatches.res1 is True) & (mismatches.res1 != mismatches.res2)
            ]
            negDf.to_csv('./output/false_negatives.csv', sep='\t', encoding='utf-8', columns=['s1', 's2'])
            posDf = mismatches[
                (not self.latin or mismatches.a1 == 'LATIN') & (not self.latin or mismatches.a2 == 'LATIN') &
                (mismatches.res1 is False) & (mismatches.res1 != mismatches.res2)
            ]
            posDf.to_csv('./output/false_positives.csv', sep='\t', encoding='utf-8', columns=['s1', 's2'])
        elif len(datasets) == 3:
            reader = pd.read_csv(getTMabsPath(datasets[0]), sep='\t',
                                 names=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"])

            res1 = pd.read_csv(
                datasets[1], sep='\t',
                names=["res1"] + list(map(lambda x: "res1_{0}".format(x), StaticValues.methods_as_saved)) + \
                      ["res1_transformed_s1", "res1_transformed_s2"]
            )
            res2 = pd.read_csv(
                datasets[2], sep='\t',
                names=["res2"] + list(map(lambda x: "res2_{0}".format(x), StaticValues.methods_as_saved)) + \
                      ["res2_transformed_s1", "res2_transformed_s2"]
            )

            mismatches = pd.concat([reader, res1, res2], axis=1)
            mismatches = mismatches.sort_values(by=['res'], ascending=False)

            for metric_name in StaticValues.methods_as_saved:
                negDf = mismatches[
                    (not self.latin or mismatches.a1 == 'LATIN') & (not self.latin or mismatches.a2 == 'LATIN') &
                    (mismatches.res1 == mismatches['res1_{0}'.format(metric_name)]) &
                    (mismatches.res2 != mismatches['res2_{0}'.format(metric_name)])
                    ]
                negDf.to_csv('./output/false_enhancedmetric_{0}.csv'.format(metric_name), sep='\t',
                             encoding='utf-8', columns=['s1', 's2', 'res', "res2_transformed_s1", "res2_transformed_s2"])

                tmpDf = mismatches[ mismatches.res1 != mismatches.res2 ]
                if not tmpDf.empty: print(tmpDf)
        else: print("Wrong number {0} of input datasets to cmp".format(len(datasets)))

    def build_dataset(self):
        build_dataset_from_geonames(only_latin=self.latin)
        filter_dataset()
