import os
from functools import partial

import click
import mlflow
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from dotenv import load_dotenv
from mlflow.models import infer_signature
from optuna.integration import CatBoostPruningCallback
from optuna.samplers import TPESampler

from src.logger import LOGGER
from src.utils import (
    catboost_calc_metrics,
    get_confusion_matrix,
)


@click.command()
@click.argument("train_path", type=click.Path(), default="./data/train.parquet")
@click.argument("val_path", type=click.Path(), default="./data/val.parquet")
@click.argument("test_path", type=click.Path(), default="./data/test.parquet")
@click.argument(
    "output_model_path", type=click.Path(), default="./models/tuned_catboost.cbm"
)
def catboost_tuning(
    train_path: str, val_path: str, test_path: str, output_model_path: str
):
    """
    Данная функция служит для тюнинга гиперпараметров CatBoostClassifier,
    его обучения на лучших параметрах, а также логирования эксперимента в mlflow.
    Сохраняются графики, относящиеся к optuna, лучшие параметры модели,
    метрики, а также сама catboost модель. Также модель попадает в mlflow model registry.

    :param train_path: путь до тренировочного датафрейма
    :param val_path: путь до валидационного датафрейма
    :param test_path: путь до тестового датафрейма
    :param output_model_path: куда локально сохранить итоговую catboost модель
    """
    LOGGER.info("catboost_tuning start")
    load_dotenv(override=True)
    mlflow.set_experiment("catboost_tuning")
    remote_server_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(remote_server_uri)

    with mlflow.start_run():
        LOGGER.info("Reading val_df")
        val_df = pd.read_parquet(val_path)
        LOGGER.info("Reading test_df")
        test_df = pd.read_parquet(test_path)
        LOGGER.info("Reading train_df")
        train_df = pd.read_parquet(train_path)
        LOGGER.info("Data is ready")

        cb_params = {
            "eval_metric": "Logloss",
            "loss_function": "Logloss",
            "task_type": "CPU",
            "border_count": 254,
            "grow_policy": "SymmetricTree",
            "boosting_type": "Plain",
            "random_seed": 7575,
            "iterations": 100,
            "auto_class_weights": "Balanced",
        }

        LOGGER.info("Creating train pool")
        train_pool = Pool(train_df.drop("target", axis=1), train_df["target"])
        LOGGER.info("Creating val pool")
        val_pool = Pool(val_df.drop("target", axis=1), val_df["target"])

        study = optuna.create_study(
            sampler=TPESampler(),
            pruner=optuna.pruners.HyperbandPruner(min_resource=1),
            direction="minimize",
        )

        good_params = {
            "learning_rate": 0.3364593306729285,
            "depth": 2,
            "l2_leaf_reg": 3.3291591295624006,
            "random_strength": 0.08204035894098327,
            "bootstrap_type": "Bernoulli",
            "min_data_in_leaf": 6,
            "subsample": 0.5062040753051366,
        }

        study.enqueue_trial(good_params)

        objective_fn_partial = partial(
            objective, train_pool=train_pool, val_pool=val_pool
        )
        LOGGER.info("Start hyperparameters tuning")
        study.optimize(objective_fn_partial, n_trials=10, show_progress_bar=True)
        LOGGER.info("Hyperparameters tuning is done")

        LOGGER.info("Creating oprimization history plot")
        plot_optimization_history_fig = optuna.visualization.plot_optimization_history(
            study
        )
        mlflow.log_figure(
            plot_optimization_history_fig, "plot_optimization_history.html"
        )

        LOGGER.info("Creating params importance plot")
        plot_param_importances_fig = optuna.visualization.plot_param_importances(study)
        mlflow.log_figure(plot_param_importances_fig, "plot_param_importances.html")

        LOGGER.info("Creating slice plot")
        plot_slice_fig = optuna.visualization.plot_slice(study)
        mlflow.log_figure(plot_slice_fig, "plot_slice.html")

        for key, value in study.best_trial.params.items():
            cb_params[key] = value

        LOGGER.info("Train final model")
        catboost_classifier = CatBoostClassifier(**cb_params)
        catboost_classifier.fit(
            train_pool,
            eval_set=val_pool,
            verbose=True,
            early_stopping_rounds=10,
            use_best_model=False,
        )
        LOGGER.info("Training is done")

        LOGGER.info("Calculating train metrics")
        _, train_metrics = catboost_calc_metrics(
            catboost_classifier, train_df.drop("target", axis=1), train_df["target"]
        )

        LOGGER.info("Calculating val metrics")
        _, val_metrics = catboost_calc_metrics(
            catboost_classifier, val_df.drop("target", axis=1), val_df["target"]
        )

        LOGGER.info("Calculating test metrics")
        test_preds, test_metrics = catboost_calc_metrics(
            catboost_classifier, test_df.drop("target", axis=1), test_df["target"]
        )

        LOGGER.info("Logging metrics into MLFlow")
        mlflow.log_params(cb_params)
        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(test_metrics)

        LOGGER.info("Calculating confusion matrix")
        fig = get_confusion_matrix(test_df["target"], test_preds)
        mlflow.log_figure(fig, "test_confusion_matrix.png")

        signature = infer_signature(test_df.drop("target", axis=1), test_preds)

        LOGGER.info("Logging the final model into MLFlow")
        mlflow.catboost.log_model(
            cb_model=catboost_classifier,
            artifact_path="tuned_catboost",
            registered_model_name="tuned_catboost",
            signature=signature,
        )

        LOGGER.info("Saving the final model locally")
        catboost_classifier.save_model(output_model_path)


def objective(trial, train_pool, val_pool):
    params = {
        "eval_metric": "Logloss",
        "loss_function": "Logloss",
        "task_type": "CPU",  # CPU because of Optuna CatBoostPruningCallback
        "border_count": 254,
        "grow_policy": "SymmetricTree",
        "boosting_type": "Plain",
        "random_seed": 7575,
        "iterations": 100,
        "auto_class_weights": "Balanced",
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 5e-1),
        "depth": trial.suggest_int("depth", 2, 4),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 5e0),
        "random_strength": trial.suggest_float("random_strength", 0, 2),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli"]
        ),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 20),
    }
    # Conditional Hyper-Parameters
    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature", 0, 10
        )
    elif params["bootstrap_type"] == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    pruning_callback = CatBoostPruningCallback(trial, "Logloss")

    catboost_classifier = CatBoostClassifier(**params)
    catboost_classifier.fit(
        train_pool,
        eval_set=val_pool,
        verbose=False,
        early_stopping_rounds=10,
        use_best_model=False,
        callbacks=[pruning_callback],
    )

    pruning_callback.check_pruned()

    score = catboost_classifier.get_best_score()
    return score["validation"]["Logloss"]


if __name__ == "__main__":
    catboost_tuning()  # pylint: disable=no-value-for-parameter
