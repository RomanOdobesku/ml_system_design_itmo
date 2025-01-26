import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def reduce_mem_usage(df):
    """
    Уменьшает размер датафрейма за счёт оптимизации типов для int и float столбцов
    :param df: входной датафрейм
    :return df: оптимизированный датафрейм
    """
    for col in df.columns:
        col_type = df[col].dtype.name

        if col_type not in [
            "object",
            "category",
            "datetime64[ns, UTC]",
            "datetime64[ns]",
            "bool",
        ]:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    return df


def get_confusion_matrix(targets, preds):
    """
    Строит нормированный confusion matrix, возвращает fig для дальнейшего логирования в mlflow
    :param targets: истинные ответы
    :param preds: предсказания модели
    :return:
    """
    cf_matrix = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(dpi=100)
    sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt=".2%", cmap="Blues")
    ax.set_title("Test Confusion Matrix \n")
    ax.set_xlabel("\nPredicted Values")
    ax.set_ylabel("Actual Values ")
    ax.xaxis.set_ticklabels(["False", "True"])
    ax.yaxis.set_ticklabels(["False", "True"])
    return fig


def catboost_calc_metrics(catboost_classifier, data, targets):
    """
    Вспомогательная функция, считает метрики классификации
    :param catboost_classifier: CatBoostClassifier
    :param data: данные в модель
    :param targets: истинные ответы
    :return: предсказания и метрики в виде словаря
    """
    proba = catboost_classifier.predict_proba(data)[:, 1]
    preds = catboost_classifier.predict(data)

    metrics = dict(
        train_logloss=log_loss(targets, proba),
        train_acc=accuracy_score(targets, preds),
        train_precision=precision_score(targets, preds),
        train_recall=recall_score(targets, preds),
        train_rocauc=roc_auc_score(targets, proba),
        train_f1=f1_score(targets, preds),
    )
    return preds, metrics
