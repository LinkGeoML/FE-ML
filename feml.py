# -*- coding: utf-8 -*-

"""Feature extraction and traditional classifiers for toponym matching.

Usage:
  feml.py [options]
  feml.py (-h | --help)
  feml.py --version

Options:
  -h --help                 show this screen.
  --version                 show version.
  -d <dataset-name>         relative path to the directory of the script being run of the dataset to use for
                            experiments. [default: dataset-string-similarity.txt].
  --permuted                permuted Jaro-Winkler metric. Default is False.
  --stemming                perform stemming. Default is False.
  --canonical               perform canonical decomposition (NFKD). Default is False.
  --sort                    sort alphanumerically.
  --ev <evaluator_type>     type of experiments to conduct. [default: SotAMetrics]
  --print                   print only computed variables. Default is False.
  --accuracyresults         store predicted results (TRUE/FALSE) in file. Default is False.
  --jobs <no>               number of CPUs utilized. [Default: 2].
  --test                    perform various test operations. Default is False.
  --ml <ML_algs>            Comma separated machine learning algorithms to run.[default: all]

Arguments:
  evaluator_type            'SotAMetrics' (default)
                            'SotAML'
                            'customFEML'
                            'DLearninng'
  ML_algs                   'rf' (Random Forest)
                            'et' (ExtraTreeClassifier)
                            'xgboost' (XGBoost)
                            'qda' (Quadratic Discriminant Analysis)
                            'lda' (Linear Discriminant Analysis)
                            'nn' (Neural Net)
                            'ada' (AdaBoost)
                            'nb' (Naive Bayes)
                            'dt' (Decision Tree)
                            'lsvm' (Linear SVM)
"""

import os, sys
import helpers
# import configparser
from docopt import docopt
from kitchen.text.converters import getwriter

import executionMethods as rc
from helpers import getTMabsPath


def main(args):
    UTF8Writer = getwriter('utf8')
    sys.stdout = UTF8Writer(sys.stdout)

    dataset_path = args['-d']

    evaluator = rc.Evaluator(args['--ml'], args['--sort'], args['--stemming'], args['--canonical'],
                             args['--permuted'], args['--print'])

    fpath_ds = getTMabsPath(dataset_path)
    if os.path.isfile(fpath_ds):
        evaluator.initialize(fpath_ds, args['--ev'], args['--jobs'], args['--accuracyresults'])
        if args['--print']: evaluator.do_the_printing()
        elif args['--test']: evaluator.test_cases(fpath_ds)
        else: evaluator.evaluate_metrics(fpath_ds)
    else: print "No file {0} exists!!!\n".format(fpath_ds)


if __name__ == "__main__":
    arguments = docopt(__doc__, version='FE-ML 0.1')
    main(arguments)
