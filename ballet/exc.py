class Error(BaseException):
    pass


class UnexpectedTravisEnvironmentError(Error):
    pass


class UnexpectedValidationStateError(Error):
    pass


class UnsuccessfulInputConversionError(Error):
    pass


class ConfigurationError(Error):
    pass


class FeatureValidationError(Error):
    pass


class FeatureRejected(FeatureValidationError):
    pass
