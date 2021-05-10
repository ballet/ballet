from ballet.validation.common import subsample_data_for_validation


def validate_feature_acceptance(accepter_class, feature, features, X_df, y_df,
                                X_df_val, y_val, subsample):
    if subsample:
        X_df, y_df, X_df_val, y_val = subsample_data_for_validation(
            X_df, y_df, X_df_val, y_val)
    accepter = accepter_class(X_df, y_df, X_df_val, y_val, features, feature)
    return accepter.judge()
