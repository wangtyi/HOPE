from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
)

def get_metrics(targets_all,
                preds_all,
                probs_all=None,
                get_report=True):
    """
    Calculate evaluation metrics for predicted labels and probabilities.

    Args:
        targets_all: True target values.
        preds_all: Predicted target values.
        probs_all: Predicted probabilities for each class (optional).
        get_report: Whether to include the classification report in the results.

    Returns:
        Dictionary containing evaluation metrics including accuracy, weighted F1, AUROC, and optionally classification report.
    """
    if len(set(targets_all)) == 2:
        roc_kwargs = {}
    else:
        roc_kwargs = {"multi_class": "ovo", "average": "macro"}
    
    acc = accuracy_score(targets_all, preds_all)
    cls_rep = classification_report(targets_all, preds_all, output_dict=True, zero_division=0)

    eval_metrics = {
        "acc": acc,
        "weighted_f1": cls_rep["weighted avg"]["f1-score"],
    }

    if get_report:
        eval_metrics["cls_report"] = cls_rep

    if probs_all is not None:
        if len(set(targets_all)) == 1:
            # Only one class present, roc_auc is not applicable, set to -1.0
            roc_auc = -1.0
        else:
            roc_auc = roc_auc_score(targets_all, probs_all, **roc_kwargs)
        eval_metrics["auroc"] = roc_auc

    return eval_metrics


