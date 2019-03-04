import os
import sys
import re

from nltk import SnowballStemmer, wordpunct_tokenize
from nltk.corpus import stopwords
from langdetect import detect, lang_detect_exception
import pycountry

from external.datasetcreator import damerau_levenshtein, jaccard, jaro, jaro_winkler, monge_elkan, cosine, \
    strike_a_match, soft_jaccard, sorted_winkler, permuted_winkler, skipgram, davies, l_jaro_winkler, lsimilarity


sys.path.append(os.path.abspath('../Toponym-Matching'))


def getTMabsPath(ds):
    return os.path.join(os.path.abspath('../Toponym-Matching'), 'dataset', ds)


def getRelativePathtoWorking(ds):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ds)


def get_langnm(s, lang_detect=False):
    lname = 'english'
    try:
        lname = pycountry.languages.get(alpha_2=detect(s)).name.lower() if lang_detect else 'english'
    except lang_detect_exception.LangDetectException as e:
        print(e)

    return lname


# Clean the string from stopwords, puctuations based on language detections feature
# Returned values #1: non-stopped words, #2: stopped words
def normalize_str(s, stop_words=None, sorting=False, lang_detect=False):
    lname = get_langnm(s, lang_detect)

    tokens = wordpunct_tokenize(s)
    words = [word.lower() for word in tokens if word.isalpha()]
    stopwords_set = set(stopwords.words(lname)) if stop_words is None else set(stop_words)

    filtered_words = sorted_nicely(filter(lambda token: token not in stopwords_set, words)) if sorting else \
        filter(lambda token: token not in stopwords_set, words)
    stopped_words = sorted_nicely(filter(lambda token: token not in filtered_words, words)) if sorting else \
        filter(lambda token: token not in filtered_words, words)

    return filtered_words, stopped_words


def perform_stemming(s, lang_detect=False):
    try:
        lname = get_langnm(s, lang_detect)

        if lname in SnowballStemmer.languages:  # See which languages are supported
            stemmer = SnowballStemmer(lname)  # Choose a language
            s = stemmer.stem(s)  # Stem a word
    except KeyError:
        pass
        # print("Unicode error for {0}\n".format(e))

    return s


def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


# def enum(*sequential, **named):
#     enums = dict(zip(sequential, range(len(sequential))), **named)
#     reverse = dict((value, key) for key, value in enums.iteritems())
#     enums['reverse_mapping'] = reverse
#     return type('Enum', (), enums)


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
        'l_jaro_winkler': l_jaro_winkler,
        'lsimilarity': lsimilarity
    }

    # The process took 33196.57 sec
    # Cosine N - grams [0.4, [0.6150486, 0.7036605891695503, 0.3975004, 0.5080190243028387]]
    # Jaro - Winkler [0.7, [0.6358974, 0.7174139942418586, 0.4484288, 0.5518910407647016]]
    # Dice bigrams [0.5, [0.6217942, 0.7536366733612166, 0.36189, 0.4889772378116455]]
    # Sorted Jaro - Winkler [0.7, [0.6189116, 0.7144158504392482, 0.3962036, 0.5097229869855858]]
    # Jaro [0.75, [0.6377728, 0.7694807735038321, 0.3933992, 0.5206265953651169]]
    # Jaccard N - grams [0.25, [0.6171748, 0.7150005798190671, 0.3896736, 0.5044326282448593]]
    # Jaro - Winkler reversed [0.75, [0.6516894, 0.7799749130854321, 0.4225876, 0.5481756216320132]]
    # Soft - Jaccard [0.6, [0.5943208, 0.6965091995193149, 0.3343116, 0.4517780491325027]]
    # Permuted Jaro - Winkler is empty
    # Damerau - Levenshtein [0.55, [0.65068, 0.7865148627208187, 0.4136332, 0.5421475393248235]]
    # Jaccard skipgrams [0.45, [0.6268896, 0.7344004693566002, 0.397558, 0.5158612965057345]]
    # Monge - Elkan [0.7, [0.595671, 0.6582955812714475, 0.397862, 0.49596916445086014]]
    # Davis and De Salles [0.65, [0.6209832, 0.7103262630715932, 0.4085916, 0.5187750663908612]]

    # The process took 38158.99 sec
    # =============================
    # Cosine N - grams[0.55, [0.6532216, 0.7929532870384532, 0.4147336, 0.5446186008887582]]
    # Jaro - Winkler[0.85, [0.6491346, 0.7978714819972373, 0.3994684, 0.53238758536534]]
    # Dice bigrams[0.55, [0.6555378, 0.7904536249887021, 0.4232876, 0.5513353438841664]]
    # Sorted Jaro - Winkler[0.85, [0.648052, 0.7972029375820032, 0.3971276, 0.5301569027211673]]
    # Jaro[0.8, [0.6608854, 0.8305803701685476, 0.4042232, 0.5437947592601909]]
    # Jaccard N - grams[0.35, [0.6550174, 0.7885240616947825, 0.423656, 0.5511771071606344]]
    # Jaro - Winkler reversed[0.75, [0.6593874, 0.7642860032751504, 0.4609308, 0.575054076922098]]
    # Soft - Jaccard[0.7, [0.64672, 0.7896897918308743, 0.3999564, 0.5309838982821329]]
    # Permuted Jaro - Winkler is empty
    # Damerau - Levenshtein[0.6, [0.663373, 0.8265968105769285, 0.4134872, 0.5512323323568598]]
    # Jaccard skipgrams[0.55, [0.658001, 0.8075493146365296, 0.414872, 0.5481409645765263]]
    # Monge - Elkan[0.85, [0.6490476, 0.8150739024555232, 0.3855756, 0.5235043338474573]]
    # Davis and De Salles[0.7, [0.6477734, 0.7471548658801334, 0.4467224, 0.5591375669786182]]

    methods = [
        ["Damerau-Levenshtein", {'orig': 0.55, 'sorted': 0.60}],
        ["Jaro", {'orig': 0.75, 'sorted': 0.8}],
        ["Jaro-Winkler", {'orig': 0.7, 'sorted': 0.85}],
        ["Jaro-Winkler reversed", {'orig': 0.75, 'sorted': 0.75}],
        ["Sorted Jaro-Winkler", {'orig': 0.7, 'sorted': 0.85}],
        ["Permuted Jaro-Winkler", {'orig': 0.7, 'sorted': 0.7}],
        ["Cosine N-grams", {'orig': 0.4, 'sorted': 0.7}],
        ["Jaccard N-grams", {'orig': 0.25, 'sorted': 0.35}],
        ["Dice bigrams", {'orig': 0.5, 'sorted': 0.55}],
        ["Jaccard skipgrams", {'orig': 0.45, 'sorted': 0.55}],
        ["Monge-Elkan", {'orig': 0.7, 'sorted': 0.85}],
        ["Soft-Jaccard", {'orig': 0.6, 'sorted': 0.7}],
        ["Davis and De Salles", {'orig': 0.65, 'sorted': 0.7}],
        ["LinkGeoML Jaro-Winkler", {'orig': 0.7, 'sorted': 0.85}],
        ["LinkGeoML Jaro-Winkler reversed", {'orig': 0.75, 'sorted': 0.75}],
        ["LinkGeoML Similarity", {'orig': 0.3, 'sorted': 0.35}],
    ]

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
        'l_jaro_winkler': 13,
        'lsimilarity': 15,
    }

    methods_as_saved = [
        "damerau_levenshtein",
        "jaccard",
        "jaro",
        "jaro_winkler",
        "jaro_winkler_reversed",
        "monge_elkan",
        "cosine",
        "strike_a_match",
        "soft_jaccard",
        "sorted_winkler",
        "skipgram",
        "davies",
        "l_jaro_winkler",
        "l_jaro_winkler_reversed",
        "lsimilarity",
    ]

    classifiers_abbr = {
        'lsvm': 0,
        'dt': 1,
        'rf': 2,
        'nn': 3,
        # 'ada': 10,
        'nb': 4,
        # 'qda': 8,
        # 'lda': 9,
        'et': 5,
        'xgboost': 6,
    }

    classifiers = [
        "Linear SVM",
        "Decision Tree", "Random Forest", "Neural Net", "Naive Bayes",
        "ExtraTreeClassifier", "XGBOOST"
        # "QDA", "LDA",
        # "AdaBoost", "Gaussian Process",
    ]

    featureColumns = [
        "Damerau-Levenshtein",
        "Jaro",
        "Jaro-Winkler",
        "Jaro-Winkler reversed",
        "Sorted Jaro-Winkler",
        # "Permuted Jaro-Winkler",
        "Cosine N-grams",
        "Jaccard N-grams",
        "Dice bigrams",
        "Jaccard skipgrams",
        "Monge-Elkan",
        "Soft-Jaccard",
        "Davis and De Salles",
        # "LinkGeoML Jaro-Winkler",
        # "LinkGeoML Jaro-Winkler reversed",
        "Damerau-Levenshtein Sorted",
        "Jaro Sorted",
        "Jaro-Winkler Sorted",
        "Jaro-Winkler reversed Sorted",
        "Sorted Jaro-Winkler Sorted",
        # "Permuted Jaro-Winkler Sorted",
        "Cosine N-grams Sorted",
        "Jaccard N-grams Sorted",
        "Dice bigrams Sorted",
        "Jaccard skipgrams Sorted",
        "Monge-Elkan Sorted",
        "Soft-Jaccard Sorted",
        "Davis and De Salles Sorted",
        # "LinkGeoML Jaro-Winkler Sorted",
        # "LinkGeoML Jaro-Winkler reversed Sorted",
        "lSimilarity_baseScore",
        "lSimilarity_mismatchScore",
        "lSimilarity_specialScore",
        "contains_str1",
        "contains_str2",
        "WordsNo_str1",
        "WordsNo_str2",
        "dashed_str1",
        "dashed_str2",
        "hasFreqTerm_str1",
        "hasFreqTerm_str2",
        "posOfHigherSim_str1_start",
        "posOfHigherSim_str1_middle",
        "posOfHigherSim_str1_end",
        "posOfHigherSim_str2_start",
        "posOfHigherSim_str2_middle",
        "posOfHigherSim_str2_end",
    ]
