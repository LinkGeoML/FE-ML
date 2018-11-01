"""Feature extraction and traditional classifiers for toponym matching.

Usage:
  feml.py [options]
  feml.py (-h | --help)
  feml.py --version

Options:
  -h --help                 show this screen.
  --version                 show version.
  -c <classifier_method>    various supported classifiers. [default: 'rf'].

Arguments:
  classifier_method:        'rf' (default)
                            'et'
                            'svm'
                            'xgboost'

"""

import helpers
import configparser
from docopt import docopt
import os, sys
import csv
import time

from featureclassifiers import evaluate_classifier
from datasetcreator import damerau_levenshtein, jaccard, jaro, jaro_winkler,monge_elkan, cosine, strike_a_match, \
    soft_jaccard, sorted_winkler, permuted_winkler, skipgram, davies


class Evaluate:
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
              ["Soft-Jaccard",0.6],
              ["Davis and De Salles", 0.65]]

    def getTMrelativePath(self, str):
        return os.path.join(os.path.abspath('../Toponym-Matching'), str)


    def prediction(self, sim_id, pred_val, real_val):
        result = ""
        var_name = ""
        if real_val == 1.0:
            if pred_val >= self.methods[sim_id - 1][1]:
                var_name = 'num_true_predicted_true'
                # result = "TRUE\tTRUE\n"
                result = "\tTRUE"
            else:
                var_name = 'num_true_predicted_false'
                # result = "TRUE\tFALSE\n"
                result = "\tFALSE"
        else:
            if pred_val >= self.methods[sim_id - 1][1]:
                var_name = 'num_false_predicted_true'
                # result = "FALSE\tTRUE\n"
                result = "\tTRUE"
            else:
                var_name = 'num_false_predicted_false'
                # result = "FALSE\tFALSE\n"
                result = "\tFALSE"

        return result, var_name


    def evaluate_metrics(self, dataset='dataset-string-similarity.txt', accuracyresults=False, results=False, permuted=False):
        num_true = 0.0
        num_false = 0.0
        num_true_predicted_true = [0.0]*len(self.methods)
        num_true_predicted_false = [0.0]*len(self.methods)
        num_false_predicted_true = [0.0]*len(self.methods)
        num_false_predicted_false = [0.0]*len(self.methods)
        timers = [0.0]*len(self.methods)
        result = {}
        file = None
        if accuracyresults:
            file = open('dataset-accuracyresults-sim-metrics.txt', 'w+')

        with open(dataset) as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                    delimiter='\t')
            for row in reader:
                if row['res'] == "TRUE":
                    num_true += 1.0
                else:
                    num_false += 1.0

        print "Reading dataset..."
        with open(dataset) as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                    delimiter='\t')

            # start_time = time.time()
            for row in reader:
                tot_res = ""
                real = 1.0 if row['res'] == "TRUE" else 0.0

                row['s1'] = row['s1'].decode('utf-8')
                row['s2'] = row['s2'].decode('utf-8')

                start_time = time.time()
                sim1 = damerau_levenshtein(row['s1'], row['s2'])
                res, varnm = self.prediction(1, sim1, real)
                timers[1 - 1] += (time.time() - start_time)
                locals()[varnm][0] += 1.0
                tot_res += res
                # if accuracyresults:
                #     file.write(res)

                start_time = time.time()
                sim8 = jaccard(row['s1'], row['s2'])
                res, varnm = self.prediction(8, sim8, real)
                timers[8 - 1] += (time.time() - start_time)
                locals()[varnm][8 - 1] += 1.0
                tot_res += res

                start_time = time.time()
                sim2 = jaro(row['s1'], row['s2'])
                res, varnm = self.prediction(2, sim2, real)
                timers[2 - 1] += (time.time() - start_time)
                locals()[varnm][2 - 1] += 1.0
                tot_res += res

                start_time = time.time()
                sim3 = jaro_winkler(row['s1'], row['s2'])
                res, varnm = self.prediction(3, sim3, real)
                timers[3 - 1] += (time.time() - start_time)
                locals()[varnm][3 - 1] += 1.0
                tot_res += res

                start_time = time.time()
                sim4 = jaro_winkler(row['s1'][::-1], row['s2'][::-1])
                res, varnm = self.prediction(4, sim4, real)
                timers[4 - 1] += (time.time() - start_time)
                locals()[varnm][4 - 1] += 1.0
                tot_res += res

                start_time = time.time()
                sim11 = monge_elkan(row['s1'], row['s2'])
                res, varnm = self.prediction(11, sim11, real)
                timers[11 - 1] += (time.time() - start_time)
                locals()[varnm][11 - 1] += 1.0
                tot_res += res

                start_time = time.time()
                sim7 = cosine(row['s1'], row['s2'])
                res, varnm = self.prediction(7, sim7, real)
                timers[7 - 1] += (time.time() - start_time)
                locals()[varnm][7 - 1] += 1.0
                tot_res += res

                start_time = time.time()
                sim9 = strike_a_match(row['s1'], row['s2'])
                res, varnm = self.prediction(9, sim9, real)
                timers[9 - 1] += (time.time() - start_time)
                locals()[varnm][9 - 1] += 1.0
                tot_res += res

                start_time = time.time()
                sim12 = soft_jaccard(row['s1'], row['s2'])
                res, varnm = self.prediction(12, sim12, real)
                timers[12 - 1] += (time.time() - start_time)
                locals()[varnm][12 - 1] += 1.0
                tot_res += res

                start_time = time.time()
                sim5 = sorted_winkler(row['s1'], row['s2'])
                res, varnm = self.prediction(5, sim5, real)
                timers[5 - 1] += (time.time() - start_time)
                locals()[varnm][5 - 1] += 1.0
                tot_res += res

                if permuted:
                    start_time = time.time()
                    sim6 = permuted_winkler(row['s1'], row['s2'])
                    res, varnm = self.prediction(6, sim6, real)
                    timers[6 - 1] += (time.time() - start_time)
                    locals()[varnm][6 - 1] += 1.0
                    tot_res += res

                start_time = time.time()
                sim10 = skipgram(row['s1'], row['s2'])
                res, varnm = self.prediction(10, sim10, real)
                timers[10 - 1] += (time.time() - start_time)
                locals()[varnm][10 - 1] += 1.0
                tot_res += res

                start_time = time.time()
                sim13 = davies(row['s1'], row['s2'])
                res, varnm = self.prediction(13, sim13, real)
                timers[13 - 1] += (time.time() - start_time)
                locals()[varnm][13 - 1] += 1.0
                tot_res += res

                if accuracyresults:
                    if real == 1.0:
                        file.write("TRUE{0}".format(tot_res + "\n"))

        if accuracyresults:
            file.close()

        for idx in range(len(self.methods)):
            try:
                timer = ( timers[idx] / float( int( num_true + num_false ) ) ) * 50000.0
                acc = ( num_true_predicted_true[idx] + num_false_predicted_false[idx] ) / ( num_true + num_false )
                pre = ( num_true_predicted_true[idx] ) / ( num_true_predicted_true[idx] + num_false_predicted_true[idx] )
                rec = ( num_true_predicted_true[idx] ) / ( num_true_predicted_true[idx] + num_true_predicted_false[idx] )
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
        #     return result


    def verifyCode(self):
        # String similarity metrics
        self.evaluate_metrics(dataset=self.getTMrelativePath('dataset/dataset-string-similarity-test.txt'),
                              accuracyresults=True, permuted=True)

        # Supervised machine learning
        evaluate_classifier(dataset=self.getTMrelativePath('dataset/dataset-string-similarity-test.txt'), method='rf',
                            accuracyresults=True, results=False)
        evaluate_classifier(dataset=self.getTMrelativePath('dataset/dataset-string-similarity-test.txt'), method='et',
                            accuracyresults=True, results=False)
        evaluate_classifier(dataset=self.getTMrelativePath('dataset/dataset-string-similarity-test.txt'), method='svm',
                            accuracyresults=True, results=False)
        evaluate_classifier(dataset=self.getTMrelativePath('dataset/dataset-string-similarity-test.txt'), method='xgboost',
                            accuracyresults=True, results=False)


def main(args):
    eval = Evaluate()
    eval.verifyCode()


if __name__ == "__main__":
    arguments = docopt(__doc__, version='FE-ML 0.1')
    main(arguments)
