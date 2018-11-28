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
  --sort                    sort alphanumerically.
  --ev <evaluator_type>     type of experiments to conduct. [default: SotAMetrics]
  --print                   print only computed variables. Default is False.
  --accuracyresults         store predicted results (TRUE/FALSE) in file. Default is False.
  --jobs <no>               number of CPUs utilized. [Default: 2].
  --testing                 perform various test operations. Default is False.

Arguments:
  evaluator_type            'SotAMetrics' (default)
                            'SotAML'
                            'customFEML'
                            'DLearninng'

"""

import os, sys
import helpers
# import configparser
from docopt import docopt
from kitchen.text.converters import getwriter

import runningCases as rc


def main(args):
    UTF8Writer = getwriter('utf8')
    sys.stdout = UTF8Writer(sys.stdout)

    dataset_path = args['-d']

    evaluator = rc.Evaluator(args['--permuted'], args['--stemming'], args['--sort'], args['--print'])

    full_dataset_path = evaluator.getTMabsPath(dataset_path)
    if os.path.isfile(full_dataset_path):
        evaluator.initialize(full_dataset_path, args['--ev'], args['--jobs'], args['--accuracyresults'])
        if args['--print']: evaluator.do_the_printing()
        else: evaluator.evaluate_metrics(full_dataset_path)
    else: print "No file {0} exists!!!\n".format(full_dataset_path)


if __name__ == "__main__":
    arguments = docopt(__doc__, version='FE-ML 0.1')
    main(arguments)
