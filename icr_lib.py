import multiprocessing

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import RandomOverSampler
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
from xgboost import XGBClassifier
from contextlib import contextmanager

class GpuQueue:

    def __init__(self, n_gpus=1):
        self.queue = multiprocessing.Manager().Queue()
        all_idxs = list(range(n_gpus)) if n_gpus > 0 else [None]
        for idx in all_idxs:
            self.queue.put(idx)

    @contextmanager
    def one_gpu_per_process(self):
        current_idx = self.queue.get()
        yield current_idx
        self.queue.put(current_idx)

def get_params_dict(trial, models, X, y):
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    # Select two models for dimensionality reduction:
    model1_to_reduce_dim_index = trial.suggest_categorical('dimensionality_reduction_model1_index',
                                                           list(range(len(models))))
    model2_to_reduce_dim_index = trial.suggest_categorical('dimensionality_reduction_model2_index',
                                                           list(range(len(models))))
    feature_names = X.columns.tolist()
    params = dict(catboost_iterations=trial.suggest_int('catboost_iterations', 50, 200),
                  catboost_learning_rate=trial.suggest_float('catboost_learning_rate', 0.01, 0.3),
                  catboost_depth=trial.suggest_int('catboost_depth', 2, 10),
                  catboost_l2_leaf_reg=trial.suggest_float('catboost_l2_leaf_reg', 1, 9),
                  # catboost_task_type='GPU',
                  # catboost_devices='(0:1)',
                  catboost_silent=True,
                  catboost_auto_class_weights='Balanced',
                  # catboost_pinned_memory_size='1gb',

                  xgboost_n_estimators=trial.suggest_int('xgboost_n_estimators', 50, 200),
                  xgboost_max_depth=trial.suggest_int('xgboost_max_depth', 2, 10),
                  xgboost_subsample=trial.suggest_float('xgboost_subsample', 0.5, 1.0),
                  xgboost_colsample_bytree=trial.suggest_float('xgboost_colsample_bytree', 0.5, 1.0),
                  xgboost_scale_pos_weight=ratio,

                  lgbm_max_depth=trial.suggest_int('lgbm_max_depth', 2, 10),
                  lgbm_learning_rate=trial.suggest_float('lgbm_learning_rate', 0.01, 0.3),
                  lgbm_num_leaves=trial.suggest_int('lgbm_num_leaves', 10, 200),
                  lgbm_colsample_bytree=trial.suggest_float('lgbm_colsample_bytree', 0.5, 1.0),
                  lgbm_is_unbalance=True,

                  tabpfn24_device='cuda',
                  tabpfn24_N_ensemble_configurations=24,
                  tabpfn24_batch_size_inference=16,

                  tabpfn64_device='cuda',
                  tabpfn64_N_ensemble_configurations=64,
                  tabpfn64_batch_size_inference=16,

                  # Adding new parameters for additional XGBoost models.
                  xgboost2_n_estimators=trial.suggest_int('xgboost2_n_estimators', 50, 200),
                  xgboost2_max_depth=trial.suggest_int('xgboost2_max_depth', 2, 10),
                  xgboost2_subsample=trial.suggest_float('xgboost2_subsample', 0.5, 1.0),
                  xgboost2_colsample_bytree=trial.suggest_float('xgboost2_colsample_bytree', 0.5, 1.0),
                  xgboost2_scale_pos_weight=ratio,

                  xgboost3_n_estimators=trial.suggest_int('xgboost3_n_estimators', 50, 200),
                  xgboost3_max_depth=trial.suggest_int('xgboost3_max_depth', 2, 10),
                  xgboost3_subsample=trial.suggest_float('xgboost3_subsample', 0.5, 1.0),
                  xgboost3_colsample_bytree=trial.suggest_float('xgboost3_colsample_bytree', 0.5, 1.0),
                  xgboost3_scale_pos_weight=ratio,
                  )

    assert all(p.split('_')[0] in models for p in params.keys())

    models_to_reduce_dim = [models[model1_to_reduce_dim_index],
                            models[model2_to_reduce_dim_index]]
    chosen_cols = _optuna_feature_select(feature_names, "", trial)
    for i, model_name in enumerate(models):
        if model_name not in ['tabpfn24', 'tabpfn64']:
            params['oversampling_' + model_name] = trial.suggest_categorical('oversampling_' + model_name,
                                                                             [True, False])

    params = update_params_dict(X, models, params, chosen_cols, feature_names, models_to_reduce_dim)
    return params


def update_params_dict(X, models, params, chosen_cols=None, feature_names=None, models_to_reduce_dim=None):
    if chosen_cols is None and feature_names is None and models_to_reduce_dim is None:
        model1_to_reduce_dim = models[params['dimensionality_reduction_model1_index']]
        model2_to_reduce_dim = models[params['dimensionality_reduction_model2_index']]
        models_to_reduce_dim = [model1_to_reduce_dim, model2_to_reduce_dim]
        feature_names = X.columns.tolist()
        chosen_cols = get_chosen_cols(params)


    for i, model_name in enumerate(models):
        if model_name in ['tabpfn24', 'tabpfn64']:
            params['oversampling_' + model_name] = True

        params['oversampling_seed_' + model_name] = i

        params['features_to_use_' + model_name] = chosen_cols if model_name in models_to_reduce_dim else feature_names
        params['gpu_needed_' + model_name] = True
        if model_name in ['lgbm', 'catboost']:
            params['gpu_needed_' + model_name] = False
    return params


def get_chosen_cols(params_dict):
    return [k for k, v in params_dict.items() if len(k) == 2 and v is True]


class AveragingClassifier():
    def __init__(self, classifiers, classifier_options, gpu_queue):
        # super().__init__()
        self.classifiers = classifiers
        self.classifier_options = classifier_options
        self.gpu_queue = gpu_queue
        assert len(self.classifiers) == len(self.classifier_options)

    def fit(self, X, y):
        for classifier, opt in zip(self.classifiers, self.classifier_options):
            X_tr, y_tr = X.copy(), y.copy()
            X_tr = X_tr[opt['features_to_use']]
            if opt['oversample']:
                X_tr, y_tr = RandomOverSampler(random_state=opt['oversampling_seed']).fit_resample(X_tr, y_tr)
            assert len(X_tr.columns) == len(set(X_tr.columns)), set(
                [x for x in X_tr.columns if list(X_tr.columns).count(x) > 1])

            if opt['gpu_needed']: # TODO: only when gpu is needed
                with self.gpu_queue.one_gpu_per_process() as gpu_id:
                    classifier.fit(X_tr, y_tr)
            else:
                classifier.fit(X_tr, y_tr)

        return self

    def predict_proba(self, X):
        # check_is_fitted(self)
        avg_pred_proba = np.mean([clf.predict_proba(X[opt['features_to_use']])
                                  for clf, opt in zip(self.classifiers, self.classifier_options)], axis=0)
        assert avg_pred_proba.shape == (len(X), 2), "The shape of avg_pred_proba is not correct"
        # "normalization" of probabilities
        n_class_0_instances = avg_pred_proba[:, 0].sum()
        n_class_1_instances = avg_pred_proba[:, 1].sum()
        # weighted probabilities based on class imbalance
        balance = np.array(
            [1 / (n_class_0_instances if i == 0 else n_class_1_instances) for i in range(avg_pred_proba.shape[1])])
        assert balance.shape == (2,), balance.shape
        normalized_proba = avg_pred_proba * balance
        normalized_proba = normalized_proba / np.sum(normalized_proba, axis=1, keepdims=1)
        return normalized_proba
    

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def preprocess_data(X, y, fill_nans, cat_encoding=False):
    # X = X.reset_index()
    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['category']).columns

    if cat_encoding:
        raise NotImplementedError()

    # Preprocessing for numerical data
    steps = []
    steps.append(('scaler', StandardScaler()))  # it is important to scale the data before knn
    if fill_nans:
        steps.append(('imputer', KNNImputer(n_neighbors=5)))
    numeric_transformer = Pipeline(steps=steps)

    # Bundle preprocessing for numerical and categorical data
    if cat_encoding:
        raise NotImplementedError()
    else:
        col_preprocessor = ColumnTransformer(
            transformers=[('num', numeric_transformer, numeric_features)])
    return col_preprocessor

def calculate_mean_epsilon(data_greeks):
    # Convert 'Epsilon' to datetime, with errors converted to NaN
    data_greeks['Epsilon'] = pd.to_datetime(data_greeks['Epsilon'], errors='coerce')
    # Compute the mean of the dates (as Unix timestamps)
    mean_date = data_greeks['Epsilon'].astype('int64').mean()
    # Fill NaNs with the mean date, and then convert back to datetime
    data_greeks['Epsilon'] = data_greeks['Epsilon'].fillna(pd.to_datetime(mean_date))
    # Convert datetime to Unix timestamp (in seconds)
    data_greeks['Epsilon'] = data_greeks['Epsilon'].astype(int) / 10 ** 9
    return data_greeks


def read_and_prepreprocess_data(for_training=True):
    # Load the data into a pandas dataframe
    if for_training:
        raw_data = pd.read_csv("train.csv")
    else:
        raw_data = pd.read_csv("test.csv")
    data_greeks = pd.read_csv("greeks.csv")
    data_greeks = data_greeks[['Id', 'Epsilon']]
    # Convert 'Epsilon' to datetime, with errors converted to NaN and calculate mean epsilon
    # Calculate mean 'Epsilon'
    data_greeks = calculate_mean_epsilon(data_greeks)
    if for_training:
        assert len(data_greeks) == len(raw_data), "Before merging DataFrames do not have the same number of rows"
    # merge dataframes on 'Id'
    data = raw_data
    if for_training:
        data = pd.merge(data, data_greeks, on='Id')
    # Define target and features
    if for_training:
        assert len(raw_data) == len(data_greeks) == len(
            data), "After merging DataFrames do not have the same number of rows"
        y = data['Class']
        X = data.drop(['Class', 'Id'], axis=1)
        submission_df = None
    else:
        y = None
        id_column = data['Id']
        X = data.drop(['Id'], axis=1)
        X['Epsilon'] = data_greeks['Epsilon'].max()
        submission_df = pd.DataFrame(id_column, columns=['Id',])

    # pre preprocessing
    numerical_columns = [column for column in X.columns if column not in ['EJ', ]]
    X[numerical_columns] = X[numerical_columns].astype('float64')
    log_cols = [column for column in X.columns if column not in ['EJ', 'BN', 'CW', 'EL', 'GL']]
    X.loc[:, log_cols] = np.log1p(X.loc[:, log_cols])
    X['BQ_is_nan'] = X.BQ.isna()  # If BQ is None, Class is always 0.
    return X, y, submission_df


def train_model(classifier, X_train, y_train):
    classifier.fit(X_train, y_train)


def get_scoring_funs():
    return {'accuracy': accuracy_score, 'log_loss': balanced_log_loss,
            'precision': precision_score, 'recall': recall_score, 'f1': f1_score}


def calc_metrics(classifier, X_test, y_test):
    scoring_funs = get_scoring_funs()
    test_scores = dict()
    y_pred_proba = classifier.predict_proba(X_test)
    assert y_pred_proba.shape == (len(X_test), 2), f"Expected shape {(len(X_test), 2)}, but got {y_pred_proba.shape}"
    y_pred = y_pred_proba.argmax(axis=1)
    assert len(y_pred) == len(X_test), f"Expected length {len(X_test)}, but got {len(y_pred)}"
    assert all(i in (0, 1) for i in y_test.unique())
    for metric in scoring_funs.keys():
        if metric == 'log_loss':
            test_scores[metric] = scoring_funs[metric](y_test, y_pred_proba)
        else:
            test_scores[metric] = scoring_funs[metric](y_test, y_pred)
    return test_scores


def balanced_log_loss(y_true, y_pred):
    nc = np.bincount(y_true)
    return log_loss(y_true, y_pred, sample_weight=1 / nc[y_true], eps=1e-15)


def fit_cv(classifier, random_state, X, y, remove_ej=False):
    cross_val_results = []
    ej = X['EJ']
    skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    for train_index, test_index in skfold.split(X, ej):  # Use 'EJ' for stratification
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        X_test_fold = X_test_fold.copy()
        if 'Epsilon' in X_train_fold.columns:
            X_test_fold['Epsilon'] = X_train_fold['Epsilon'].max()
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        classifier.fit(X_train_fold, y_train_fold)
        y_pred = classifier.predict(X_test_fold)
        assert len(y_pred) == len(X_test_fold)
        fold_results = calc_metrics(classifier, X_test_fold, y_test_fold)
        cross_val_results.append(fold_results)
    test_scores = dict()
    return cross_val_results, test_scores


def _optuna_feature_select(feature_names, suggest_suffix, trial):
    return [f for f in feature_names if trial.suggest_categorical(f + suggest_suffix, [True, False])]


def get_classifier(model_name, params):
    match model_name:
        case 'catboost':
            return CatBoostClassifier(
                **{k[len(model_name) + 1:]: v for k, v in params.items() if k.startswith(model_name)})
        case 'xgboost' | 'xgboost2' | 'xgboost3':
            return XGBClassifier(**{k[len(model_name) + 1:]: v for k, v in params.items() if k.startswith(model_name)})
        case 'lgbm':
            return LGBMClassifier(**{k[len(model_name) + 1:]: v for k, v in params.items() if k.startswith(model_name)})
        case 'tabpfn24' | 'tabpfn64':
            return TabPFNClassifier(
                **{k[len(model_name) + 1:]: v for k, v in params.items() if k.startswith(model_name)})
        case _:
            raise ValueError("There is no classifier for the given model name.")


def get_pipeline(models_list, params, X, y, gpu_queue):
    pipelines = dict()
    for model_name in models_list:
        print(f"Building pipeline for: {model_name}")

        pipelines[model_name] = Pipeline(
            steps=[('preprocessor',
                    preprocess_data(X[params['features_to_use_' + model_name]], y, fill_nans=False)),
                   (model_name, get_classifier(model_name, params))]
        )
    classifier_options = []
    for model_name in models_list:
        classifier_options.append(dict(
            features_to_use=params['features_to_use_' + model_name],
            oversample=params['oversampling_' + model_name],
            oversampling_seed=params['oversampling_seed_' + model_name],
            gpu_needed=params['gpu_needed_' + model_name]
        ))

    clf = AveragingClassifier(list(pipelines.values()), classifier_options, gpu_queue)
    averaging_pipeline = Pipeline(steps=[('averaging', clf)])
    return averaging_pipeline
