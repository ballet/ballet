from ballet.validation.common import subsample_data_for_validation


def validate_feature_acceptance(accepter_class, feature, features, X_df, y_df,
                                y, subsample):
    if subsample:
        X_df, y_df, y = subsample_data_for_validation(X_df, y_df, y)
    accepter = accepter_class(X_df, y_df, y, features, feature)
    return accepter.judge()
