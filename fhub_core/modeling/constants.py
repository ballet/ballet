RANDOM_STATE = 1754

def _make_multiclass(lst):
    return [
        l + '_' + MULTICLASS_METRIC_AGGREGATION
        for l in lst
    ]


MULTICLASS_METRIC_AGGREGATION = "micro"

CLASSIFICATION_SCORING = ['accuracy', 'roc_auc']

BINARY_CLASSIFICATION_SCORING = (
    CLASSIFICATION_SCORING + ['precision', 'recall'])

MULTICLASS_CLASSIFICATION_SCORING = (
    CLASSIFICATION_SCORING + _make_multiclass(['precision', 'recall'])
)

REGRESSION_SCORING = ['neg_mean_squared_error', 'r2']

SCORING_NAME_MAPPER = {
    # classification
    'accuracy': 'Accuracy',
    'average_precision': 'Average Precision',
    'f1': 'F1 Score (Binary)',
    'f1_micro': 'F1 Score (micro-averaged)',
    'f1_macro': 'F1 Score (macro-averaged)',
    'f1_weighted': 'F1 Score (weighted average)',
    'f1_samples': 'F1 Score (by multilabel sample)',
    'neg_log_loss': 'Negative Log Loss',
    'precision': 'Precision (Binary)',
    'precision_micro': 'Precision (micro-averaged)',
    'precision_macro': 'Precision (macro-averaged)',
    'precision_weighted': 'Precision (weighted average)',
    'precision_samples': 'Precision (by multilabel sample)',
    'recall': 'Recall (Binary)',
    'recall_micro': 'Recall (micro-averaged)',
    'recall_macro': 'Recall (macro-averaged)',
    'recall_weighted': 'Recall (weighted average)',
    'recall_samples': 'Recall (by multilabel sample)',
    'roc_auc': 'ROC AUC Score',

    # regression
    'explained_variance': 'Explained Variance',
    'neg_mean_absolute_error': 'Negative Mean Absolute Error',
    'neg_mean_squared_error': 'Negative Mean Squared Error',
    'neg_mean_squared_lod_error': 'Negative Mean Squared Log Error',
    'neg_median_absolute_error': 'Negative Median Absolute Error',
    'r2': 'R-squared',
}



