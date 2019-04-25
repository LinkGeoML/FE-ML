import pandas as pd
import numpy as np
import config

from helpers import StaticValues
from femlAlgorithms import transform
from external.datasetcreator import LSimilarityVars, lsimilarity_terms, score_per_term, weighted_terms


class Features:
    fields = [
        "s1",
        "s2",
        "status",
        "gid1",
        "gid2",
        "alphabet1",
        "alphabet2",
        "alpha2_cc1",
        "alpha2_cc2",
    ]

    dtypes = {
        's1': str, 's2': str,
        'status': str,
        'gid1': np.int32, 'gid2': np.int32,
        'alphabet1': str, 'alphabet1': str,
        'alpha2_cc1': str, 'alpha2_cc2': str
    }

    def __init__(self):
        self.clf_method = config.initialConfig.classification_method
        self.data_df = None

    def load_data(self, fname):
        self.data_df = pd.read_csv(fname, sep='\t', names=self.fields, dtype=self.dtypes, na_filter=False)

    def build_features(self):
        y = self.data_df['status'].values

        fX = None
        if self.clf_method.lower() == 'basic':
            X = self.data_df.apply(lambda row: transform(row['s1'], row['s2']), axis=1)
            fX = X.apply(lambda row: pd.Series([
                StaticValues.algorithms['damerau_levenshtein'](*row),
                StaticValues.algorithms['jaro'](*row),
                StaticValues.algorithms['jaro_winkler'](*row),
                StaticValues.algorithms['jaro_winkler'](row[0][::-1], row[1][::-1]),
                StaticValues.algorithms['sorted_winkler'](*row),
                StaticValues.algorithms['cosine'](*row),
                StaticValues.algorithms['jaccard'](*row),
                StaticValues.algorithms['strike_a_match'](*row),
                StaticValues.algorithms['skipgram'](*row),
                StaticValues.algorithms['monge_elkan'](*row),
                StaticValues.algorithms['soft_jaccard'](*row),
                StaticValues.algorithms['davies'](*row),
            ], index=StaticValues.featureColumns[0:12])).values
        elif self.clf_method.lower() == 'basic_sorted':
            X = self.data_df.apply(lambda row: transform(row['s1'], row['s2']), axis=1)
            fX1 = X.apply(lambda row: pd.Series([
                StaticValues.algorithms['damerau_levenshtein'](*row),
                StaticValues.algorithms['jaro'](*row),
                StaticValues.algorithms['jaro_winkler'](*row),
                StaticValues.algorithms['jaro_winkler'](row[0][::-1], row[1][::-1]),
                StaticValues.algorithms['sorted_winkler'](*row),
                StaticValues.algorithms['cosine'](*row),
                StaticValues.algorithms['jaccard'](*row),
                StaticValues.algorithms['strike_a_match'](*row),
                StaticValues.algorithms['skipgram'](*row),
                StaticValues.algorithms['monge_elkan'](*row),
                StaticValues.algorithms['soft_jaccard'](*row),
                StaticValues.algorithms['davies'](*row),
            ], index=StaticValues.featureColumns[0:12]))

            X_sorted = self.data_df.apply(
                lambda row: transform(row['s1'], row['s2'], sorting=True, canonical=True), axis=1)
            fX2 = X_sorted.apply(lambda row: pd.Series([
                StaticValues.algorithms['damerau_levenshtein'](*row),
                StaticValues.algorithms['jaro'](*row),
                StaticValues.algorithms['jaro_winkler'](*row),
                StaticValues.algorithms['jaro_winkler'](row[0][::-1], row[1][::-1]),
                StaticValues.algorithms['cosine'](*row),
                StaticValues.algorithms['jaccard'](*row),
                StaticValues.algorithms['strike_a_match'](*row),
                StaticValues.algorithms['skipgram'](*row),
                StaticValues.algorithms['monge_elkan'](*row),
                StaticValues.algorithms['soft_jaccard'](*row),
                StaticValues.algorithms['davies'](*row),
                StaticValues.algorithms['l_jaro_winkler'](*row),
                StaticValues.algorithms['l_jaro_winkler'](row[0][::-1], row[1][::-1]),
            ], index=StaticValues.featureColumns[12:25]))

            fX = pd.concat([fX1, fX2], axis=1).values
        else:  # lgm
            X = self.data_df.apply(lambda row: transform(row['s1'], row['s2']), axis=1)
            fX1 = X.apply(lambda row: pd.Series([
                StaticValues.algorithms['damerau_levenshtein'](*row),
                StaticValues.algorithms['jaro'](*row),
                StaticValues.algorithms['jaro_winkler'](*row),
                StaticValues.algorithms['jaro_winkler'](row[0][::-1], row[1][::-1]),
                StaticValues.algorithms['sorted_winkler'](*row),
                StaticValues.algorithms['cosine'](*row),
                StaticValues.algorithms['jaccard'](*row),
                StaticValues.algorithms['strike_a_match'](*row),
                StaticValues.algorithms['skipgram'](*row),
                StaticValues.algorithms['monge_elkan'](*row),
                StaticValues.algorithms['soft_jaccard'](*row),
                StaticValues.algorithms['davies'](*row),
            ], index=StaticValues.featureColumns[0:12]))

            X_sorted = self.data_df.apply(
                lambda row: transform(row['s1'], row['s2'], sorting=True, canonical=True), axis=1)
            fX2 = X_sorted.apply(lambda row: pd.Series([
                StaticValues.algorithms['damerau_levenshtein'](*row),
                StaticValues.algorithms['jaro'](*row),
                StaticValues.algorithms['jaro_winkler'](*row),
                StaticValues.algorithms['jaro_winkler'](row[0][::-1], row[1][::-1]),
                StaticValues.algorithms['cosine'](*row),
                StaticValues.algorithms['jaccard'](*row),
                StaticValues.algorithms['strike_a_match'](*row),
                StaticValues.algorithms['skipgram'](*row),
                StaticValues.algorithms['monge_elkan'](*row),
                StaticValues.algorithms['soft_jaccard'](*row),
                StaticValues.algorithms['davies'](*row),
                StaticValues.algorithms['l_jaro_winkler'](*row),
                StaticValues.algorithms['l_jaro_winkler'](row[0][::-1], row[1][::-1]),
                self._compute_lsimilarity(row[0], row[1], 'damerau_levenshtein'),
                self._compute_lsimilarity(row[0], row[1], 'davies'),
                self._compute_lsimilarity(row[0], row[1], 'skipgram'),
                self._compute_lsimilarity(row[0], row[1], 'soft_jaccard'),
                self._compute_lsimilarity(row[0], row[1], 'strike_a_match'),
                self._compute_lsimilarity(row[0], row[1], 'cosine'),
                self._compute_lsimilarity(row[0], row[1], 'jaccard'),
                self._compute_lsimilarity(row[0], row[1], 'monge_elkan'),
                self._compute_lsimilarity(row[0], row[1], 'jaro_winkler'),
                self._compute_lsimilarity(row[0], row[1], 'jaro'),
                self._compute_lsimilarity(row[0], row[1], 'jaro_winkler_r'),
                self._compute_lsimilarity(row[0], row[1], 'l_jaro_winkler'),
                self._compute_lsimilarity(row[0], row[1], 'l_jaro_winkler_r'),
            ] + list(self._compute_lsimilarity_base_scores(row[0], row[1], 'damerau_levenshtein')),
                index=StaticValues.featureColumns[12:41]))

            fX = pd.concat([fX1, fX2], axis=1).values

        return fX, y

    @staticmethod
    def _compute_lsimilarity(s1, s2, metric, w_type='avg'):
        baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
            s1, s2, LSimilarityVars.per_metric_optimal_values[metric][w_type][0])

        if metric in ['jaro_winkler_r', 'l_jaro_winkler_r']:
            return weighted_terms(
                {'a': [x[::-1] for x in baseTerms['a']], 'b': [x[::-1] for x in baseTerms['b']],
                 'len': baseTerms['len'], 'char_len': baseTerms['char_len']},
                {'a': [x[::-1] for x in mismatchTerms['a']], 'b': [x[::-1] for x in mismatchTerms['b']],
                 'len': mismatchTerms['len'], 'char_len': mismatchTerms['char_len']},
                {'a': [x[::-1] for x in specialTerms['a']], 'b': [x[::-1] for x in specialTerms['b']],
                 'len': specialTerms['len'], 'char_len': specialTerms['char_len']},
                metric[:-2], True if w_type == 'avg' else False
            )
        else:
            return weighted_terms(baseTerms, mismatchTerms, specialTerms, metric, True if w_type == 'avg' else False)

    @staticmethod
    def _compute_lsimilarity_base_scores(s1, s2, metric, w_type='avg'):
        baseTerms, mismatchTerms, specialTerms = lsimilarity_terms(
            s1, s2, LSimilarityVars.per_metric_optimal_values[metric][w_type][0])
        return score_per_term(baseTerms, mismatchTerms, specialTerms, metric)
