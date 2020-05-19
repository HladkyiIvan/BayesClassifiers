from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import numpy as np
import pandas as pd


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


class MultidimensionalBayesClassifier:
    def __init__(self, desc_is_linear=False):
        if not isinstance(desc_is_linear, bool):
            raise TypeError('is_linear should have float boolean.')

        self.__classes_info = {}
        self.__classes_info['classes_priory'] = []
        self.__desc_is_linear = desc_is_linear

    def fit(self, X, y):
        self.__check_dataframe(X, 'X_train')
        self.__check_dataframe(y, 'y_train')

        self.__data_len = len(y)
        self.__classes = np.sort(np.unique(y))
        self.__num_classes = len(self.__classes)

        self.__set_classes_priory(y)

        for feat in X.columns:
            self.__set_feature_info(feat, X[feat].values, np.squeeze(y.values))

    def __set_classes_priory(self, y):
        for cl in self.__classes:
            self.__classes_info['classes_priory'].append(np.sum(y.values == cl) / self.__data_len)

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

