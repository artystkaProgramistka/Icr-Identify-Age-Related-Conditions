
import pandas as pd
from icr_lib import read_and_prepreprocess_data, fit_cv, get_pipeline, get_params_dict, GpuQueue
import optuna
from optuna.visualization import plot_optimization_history
from pathlib import Path
import pickle
import plotly
from functools import partial
from sklearn.utils import shuffle


def objective(trial, gpu_queue):
    X, y, _ = read_and_prepreprocess_data()
    X, y = shuffle(X, y, random_state=42)

    models_list = ['catboost', 'xgboost', 'xgboost2', 'xgboost3', 'lgbm', 'tabpfn24', 'tabpfn64']
    # models_list = ['catboost', 'xgboost', 'tabpfn24', 'tabpfn64']

    params = get_params_dict(trial, models_list, X, y)

    averaging_pipeline = get_pipeline(models_list, params, X, y, gpu_queue)

    cv_results, test_scores = fit_cv(averaging_pipeline, 42, X, y)
    return pd.DataFrame(cv_results).log_loss.mean()


STUDY_BACKUP_FILE = Path("study_backup.pkl")


def save(study, iter):
    pickle.dump(study, STUDY_BACKUP_FILE.open('wb'))


STUDY_BACKUP_FILE = Path("study_backup.pkl")


def plot_study_results(study):
    fig = plot_optimization_history(study)
    plotly.io.write_image(fig, 'optimization_plot.png')


def get_best_model_params():
    if STUDY_BACKUP_FILE.is_file():
        study = pickle.load(STUDY_BACKUP_FILE.open('rb'))
        best_trial = study.best_trial
        print(f"Parameters of the best model (Trial {best_trial.number}):")
        for key, value in best_trial.params.items():
            print(f"{key}: {value}")
        print(f"Best score: {best_trial.value}")
        return best_trial.params
    else:
        print("Study file not found.")
        return None


def get_list_of_columns(input_string):
    lines = input_string.strip().split("\n")
    columns = []
    for line in lines:
        column, value = line.strip().split(":")
        if value.strip() == 'True':
            columns.append(column.strip())
    return columns


def main():
    study = optuna.create_study(direction='minimize')
    if STUDY_BACKUP_FILE.is_file():
        study = pickle.load(STUDY_BACKUP_FILE.open('rb'))
        print('BEST OPTUNA PARAMS:', study.best_params)
        print('BEST SCORE:', study.best_value)
        X_train, y_train, _ = read_and_prepreprocess_data()
        models_list = ['catboost', 'xgboost', 'xgboost2', 'xgboost3', 'lgbm', 'tabpfn24', 'tabpfn64']
        # models_list = ['catboost', 'xgboost', 'tabpfn24', 'tabpfn64']
        print('BEST PARAMS', get_params_dict(study.best_trial, models_list, X_train, y_train))
    gpu_queue = GpuQueue(1)
    study.optimize(partial(objective, gpu_queue=gpu_queue), n_trials=800, callbacks=[save], n_jobs=1)
    print(study.best_params)


if __name__ == '__main__':
    main()
