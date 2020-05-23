from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF


class NaiveBayesClassifier:
    def __init__(self, smoothing_alpha=0.01, kernel=None, bandwidth=2.):
        if not isinstance(smoothing_alpha, float):
            raise TypeError('smoothing_alpha should have float type.')
        if not isinstance(kernel, str) and kernel != None:
            raise TypeError('kernel should have str type.')
        if not isinstance(bandwidth, float):
            raise TypeError('bandwidth should have float type.')

        assert smoothing_alpha > 0., 'smoothing_alpha should be higher than 0.'
        assert kernel in [None, 'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear',
                          'cosine'], "kernel should be None or one of 'gaussian'|'tophat'|'epanechnikov'|'exponential'|linear'|'cosine'"
        assert bandwidth > 0., 'bandwidth should be higher than 0.'

        self.__classes_priory = []
        self.__features_info = {}
        self.__smoothing_alpha = smoothing_alpha
        self.__kernel = kernel
        self.__bandwidth = bandwidth

    def fit(self, X, y):
        self.__check_dataframe(X, 'X_train')
        self.__check_dataframe(y, 'y_train')

        self.__data_len = len(y)
        self.__classes = np.sort(np.unique(y))
        self.__num_classes = len(self.__classes)
        self.__categorical_threshold = round(np.sqrt(self.__data_len * 0.5))  # empirical formula

        self.__set_classes_priory(y)

        for feat in X.columns:
            self.__set_feature_info(feat, X[feat].values, np.squeeze(y.values))

    def __set_classes_priory(self, y):
        for cl in self.__classes:
            self.__classes_priory.append(np.sum(y.values == cl) / self.__data_len)

    def __set_feature_info(self, feature_name, feature_col, target_col):
        self.__features_info[feature_name] = {}
        feat_type = ''
        feat_unique_count = len(np.unique(feature_col))

        if feat_unique_count < self.__categorical_threshold:
            feat_type = 'discrete'
            self.__features_info[feature_name]['categories_proba'] = {}
            self.__features_info[feature_name]['num_categories'] = feat_unique_count

            for category in np.sort(np.unique(feature_col)):
                self.__features_info[feature_name]['categories_proba'][category] = []

                for cl in self.__classes:
                    self.__features_info[feature_name]['categories_proba'][category].append(
                        self.__get_category_proba_for_class(feature_col[target_col == cl], category, feat_unique_count))
        else:
            feat_type = 'continuous'
            self.__features_info[feature_name]['dist'] = []

            for cl in self.__classes:
                self.__features_info[feature_name]['dist'].append(
                    self.__get_continuous_feature_parameters(feature_col[target_col == cl]))

        self.__features_info[feature_name]['type'] = feat_type

    def __get_category_proba_for_class(self, class_feature_col, category, feat_unique_count):
        numerator = np.sum(class_feature_col == category) + self.__smoothing_alpha
        # ---------  ----------------------------------------------------------
        denominator = class_feature_col.shape[0] + feat_unique_count * self.__smoothing_alpha

        smoothed_proba = numerator / denominator
        return smoothed_proba

    def __get_continuous_feature_parameters(self, class_feature_col):
        if self.__kernel:
            dist = KernelDensity(bandwidth=self.__bandwidth, kernel=self.__kernel)
            dist.fit(class_feature_col.reshape(len(class_feature_col), 1))
        else:
            mu = np.mean(class_feature_col)
            sigma = np.std(class_feature_col)
            dist = norm(mu, sigma)
        return dist

    def predict(self, data):
        self.__check_dataframe(data, 'dataset')
        return data.apply(lambda row: self.__get_single_prediction(row, 'class'), axis=1).to_numpy()

    def predict_proba(self, data, normalize=True):
        self.__check_dataframe(data, 'dataset')
        predictions = np.matrix(data.apply(lambda row: self.__get_single_prediction(row, 'proba'), axis=1).tolist())

        if normalize:
            return np.apply_along_axis(self.__normalize_proba, 1, predictions)

        return predictions

    def __normalize_proba(self, pred):
        pred_sum = np.sum(pred)

        for i in range(len(pred)):
            pred[i] /= pred_sum

        return pred

    def predict_log(self, data):
        self.__check_dataframe(data, 'dataset')
        return np.matrix(data.apply(lambda row: self.__get_single_prediction(row, 'log'), axis=1).tolist())

    def __get_single_prediction(self, example, pred_type='proba'):
        results = []
        for i in range(len(self.__classes)):
            if pred_type == 'log':
                results.append(np.log(self.__classes_priory[i]))
            else:
                results.append(self.__classes_priory[i])

            for key, value in self.__features_info.items():
                feature_proba = 0

                if value['type'] == 'discrete':
                    feature_proba = value['categories_proba'][example[key]][i]
                elif value['type'] == 'continuous':
                    if self.__kernel:
                        feature_proba = np.exp(
                            value['dist'][i].score_samples(np.float64(example[key]).reshape(1, -1))[0])
                    else:
                        feature_proba = value['dist'][i].pdf(example[key])

                if pred_type == 'log':
                    results[i] += np.log(feature_proba)
                else:
                    results[i] *= feature_proba

        if pred_type == 'class':
            return np.argmax(results)

        return results

    def get_params(self):
        return {'data_len': self.__data_len,
                'classes_priory': self.__classes_priory,
                'smoothing_alpha': self.__smoothing_alpha,
                'kernel': self.__kernel,
                'bandwidth': self.__bandwidth,
                'categorical_threshold': self.__categorical_threshold,
                'features_info': self.__features_info}

    def __check_dataframe(self, df, df_name):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{df_name} should have type pd.DataFrame.")

        assert not df.isnull().values.any(), f"{df_name} should not contain NaN values."

        for column in df.columns:
            assert not pd.api.types.infer_dtype(df[column]).startswith(
                'mixed'), f"Column '{column}' of {df_name} should not contain mixed type of values."


class BayesClassifier:
    def __init__(self, descr_is_linear=False):
        if not isinstance(descr_is_linear, bool):
            raise TypeError('is_linear should have float boolean.')

        self.__classes_info = {}
        self.__descr_is_linear = descr_is_linear

    def fit(self, X, y):
        self.__check_dataframe(X, 'X_train')
        self.__check_dataframe(y, 'y_train')
        self.__data_len = len(y)
        self.__classes = np.sort(np.unique(y))
        self.__num_classes = len(self.__classes)

        for cl in self.__classes:
            self.__classes_info[cl] = {}
            self.__classes_info[cl]['size'] = sum(y.values == cl)
            self.__classes_info[cl]['priory'] = self.__classes_info[cl]['size'] / self.__data_len
            self.__classes_info[cl]['mean'] = np.mean(X[y.values == cl].values, axis=0)

            if not self.__descr_is_linear:
                subtraction = X[y.values == cl].values - self.__classes_info[cl]['mean']
                dot = np.apply_along_axis(lambda row: np.dot(row[np.newaxis].T, row[np.newaxis]), axis=1,
                                          arr=subtraction)
                self.__classes_info[cl]['covariation_matrix'] = np.mean(dot, axis=0)

        if self.__descr_is_linear:
            means = np.zeros((self.__data_len, len(X.columns)))

            for i in range(self.__data_len):
                means[i:] = self.__classes_info[y.iloc[i, 0]]['mean']

            subtraction = X.values - means
            dot = np.apply_along_axis(lambda row: np.dot(row[np.newaxis].T, row[np.newaxis]), axis=1, arr=subtraction)
            self.__classes_info['covariation_matrix'] = np.mean(dot, axis=0)

    def predict(self, data):
        self.__check_dataframe(data, 'dataset')
        return data.apply(lambda row: self.__get_single_prediction(row.to_numpy()), axis=1).to_numpy()

    def __get_single_prediction(self, example):
        results = []

        if self.__descr_is_linear:
            inv_cov = np.linalg.inv(self.__classes_info['covariation_matrix'])

        for cl in self.__classes:
            if not self.__descr_is_linear:
                diff = example - self.__classes_info[cl]['mean']
                inv_cov = np.linalg.inv(self.__classes_info[cl]['covariation_matrix'])

                first_part = np.squeeze(np.dot(np.dot(diff[np.newaxis], inv_cov), diff[np.newaxis].T)) / 2
                second_part = np.log(np.linalg.det(self.__classes_info[cl]['covariation_matrix'])) / 2
                results.append(np.log(self.__classes_info[cl]['priory']) - first_part - second_part)
            else:
                mult = np.squeeze(np.dot(inv_cov, self.__classes_info[cl]['mean'][np.newaxis].T))
                first_part = np.squeeze(np.dot(self.__classes_info[cl]['mean'][np.newaxis], mult)) / 2
                second_part = np.dot(example, mult)
                results.append(np.log(self.__classes_info[cl]['priory']) - first_part + second_part)

        return np.argmax(results)

    def get_params(self):
        return {'data_len': self.__data_len,
                'descr_is_linear': self.__descr_is_linear,
                'classes_info': self.__classes_info}

    def __check_dataframe(self, df, df_name):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{df_name} should have type pd.DataFrame.")

        assert not df.isnull().values.any(), f"{df_name} should not contain NaN values."

        for column in df.columns:
            assert not pd.api.types.infer_dtype(df[column]).startswith(
                'mixed'), f"Column '{column}' of {df_name} should not contain mixed type of values."


class SemiparametricBayesClassifier:
    def __init__(self, corr_type='pearson', kernel='gaussian', bandwidth=2.):
        if corr_type not in ['pearson', 'spearman']:
            raise Exception("corr_type should be one of ['pearson', 'spearman']")
        if not isinstance(kernel, str) and kernel != None:
            raise TypeError('kernel should have str type.')
        if not isinstance(bandwidth, float):
            raise TypeError('bandwidth should have float type.')

        assert kernel in [None, 'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear',
                          'cosine'], "kernel should be None or one of 'gaussian'|'tophat'|'epanechnikov'|'exponential'|linear'|'cosine'"
        assert bandwidth > 0., 'bandwidth should be higher than 0.'

        self.__classes_info = {}
        self.__corr_type = corr_type
        self.__kernel = kernel
        self.__bandwidth = bandwidth

    def fit(self, X, y):
        self.__check_dataframe(X, 'X_train')
        self.__check_dataframe(y, 'y_train')
        self.__data_len = len(y)
        self.__classes = np.sort(np.unique(y))
        self.__num_classes = len(self.__classes)

        for cl in self.__classes:
            self.__classes_info[cl] = {}
            self.__classes_info[cl]['priory'] = X[y.values == cl].shape[0] / self.__data_len
            self.__classes_info[cl]['corr_matrix'] = X[y.values == cl].corr(self.__corr_type)
            feature_dict = {}

            for feat in X.columns:
                feature_dict[feat] = {}
                class_feature_data = X[y.values == cl][feat].values
                feature_dict[feat]['ECDF'] = ECDF(class_feature_data)

                dist = KernelDensity(bandwidth=self.__bandwidth, kernel=self.__kernel)
                dist.fit(class_feature_data.reshape(len(class_feature_data), 1))
                feature_dict[feat]['dist'] = dist

            self.__classes_info[cl]['features'] = feature_dict

    def __compute_copula(self, gamma, corr_matrix):
        subtraction = np.linalg.inv(corr_matrix) - np.identity(corr_matrix.shape[0])
        degree = np.squeeze(np.dot(gamma[np.newaxis], np.dot(subtraction, gamma[np.newaxis].T))) / -2
        copula = np.exp(degree) / np.sqrt(np.linalg.det(corr_matrix))
        return copula

    def predict(self, data):
        self.__check_dataframe(data, 'dataset')
        return data.apply(lambda row: self.__get_single_prediction(row), axis=1).to_numpy()

    def __get_single_prediction(self, example):
        results = []

        for cl in self.__classes:
            ECDFs = []
            prob_mults = 1
            for feat, value in example.items():
                ECDFs.append(self.__classes_info[cl]['features'][feat]['ECDF'](value))
                prob_mults *= np.exp(
                    self.__classes_info[cl]['features'][feat]['dist'].score_samples(np.float64(value).reshape(1, -1))[
                        0])

            copula = self.__compute_copula(np.array(ECDFs), self.__classes_info[cl]['corr_matrix'])

            results.append(copula * prob_mults * self.__classes_info[cl]['priory'])

        return np.argmax(results)

    def get_params(self):
        return {'data_len': self.__data_len,
                'corr_type': self.__corr_type,
                'kernel': self.__kernel,
                'bandwidth': self.__bandwidth,
                'classes_info': self.__classes_info}

    def __check_dataframe(self, df, df_name):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{df_name} should have type pd.DataFrame.")

        assert not df.isnull().values.any(), f"{df_name} should not contain NaN values."

        for column in df.columns:
            assert not pd.api.types.infer_dtype(df[column]).startswith(
                'mixed'), f"Column '{column}' of {df_name} should not contain mixed type of values."


