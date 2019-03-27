# -*- coding: utf-8 -*-

"""Feature extraction and traditional classifiers for toponym matching.

Usage:
  feml.py [options]
  feml.py (-h | --help)
  feml.py --version

Options:
  -h --help                     show this screen.
  --version                     show version.
  -d <dataset-name>             relative path to the directory of the script being run of the dataset to use for
                                experiments. [default: dataset-string-similarity.txt].
  --permuted                    permuted Jaro-Winkler metric. Default is False.
  --stemming                    perform stemming. Default is False.
  --canonical                   perform canonical decomposition (NFKD). Default is False.
  --sort                        sort alphanumerically.
  --ev <evaluator_type>         type of experiments to conduct. [default: SotAMetrics]
  --accuracyresults             store predicted results (TRUE/FALSE) in file. Default is False.
  --jobs <no>                   number of CPUs utilized. [Default: 2].
  --test <no>                   perform various test operations. [default: 0].
  --ml <ML_algs>                Comma separated machine learning algorithms to run. [default: all]
  --cmp                         Print output results that a comparison produces. Default is False.
  --onlyLATIN                   Check for similarities only both strings use LATIN chars. Default is False.
  --optimalThres                Find best threshold for metrics.
  --optimalSortingThres         Find best threshold for metric used to decide whether to apply sorting or not.
  --buildDataset                Build the dataset for evaluation.
  --fs <FeatureSelection>       Method to use for feature selection. [default: sfm]

Arguments:
  evaluator_type            'SotAMetrics' (default)
                            'SotAML'
                            'customFEML'
                            'DLearninng'
                            'TestMetrics'
                            'customFEMLExtended'
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
  FeatureSelection          'sfm' (default)
                            'rfe'
"""

import os, sys
# import configparser
from docopt import docopt
from kitchen.text.converters import getwriter

import executionMethods as rc
from helpers import getTMabsPath


def main(args):
    UTF8Writer = getwriter('utf8')
    sys.stdout = UTF8Writer(sys.stdout)

    dataset_path = [x for x in args['-d'].split(',')]

    evaluator = rc.Evaluator(
        args['--ml'], args['--sort'], args['--stemming'], args['--canonical'], args['--permuted'], args['--onlyLATIN']
    )

    if args['--buildDataset']:
        evaluator.build_dataset()
        sys.exit(0)

    fpath_ds = getTMabsPath(dataset_path[0])
    if os.path.isfile(fpath_ds):
        evaluator.initialize(fpath_ds, args['--ev'], args['--jobs'], args['--accuracyresults'])

        if int(args['--test']): evaluator.test_cases(fpath_ds, int(args['--test']))
        elif args['--cmp']: evaluator.print_false_posneg(dataset_path)
        elif args['--optimalThres']: evaluator.evaluate_metrics_with_various_thres(fpath_ds)
        elif args['--optimalSortingThres']: evaluator.evaluate_sorting_with_various_thres(fpath_ds)
        else: evaluator.evaluate_metrics(fpath_ds, args['--fs'])
    else: print("No file {0} exists!!!\n".format(fpath_ds))


if __name__ == "__main__":
    arguments = docopt(__doc__, version='FE-ML 0.2')
    main(arguments)
