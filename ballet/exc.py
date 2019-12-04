class BalletWarning(UserWarning):
    """Base warning for ballet"""
    pass


class BalletError(Exception):
    """Base error for ballet"""
    pass


class UnexpectedTravisEnvironmentError(BalletError):
    """The environment within Travis CI testing was unexpected"""
    pass


class UnsuccessfulInputConversionError(BalletError):
    """Input-type conversion for execution within pipeline was unsuccessful"""
    pass


class ConfigurationError(BalletError):
    """BalletError in configuration of the ballet project"""
    pass


class FeatureValidationError(BalletError):
    """Base error for feature validation routine"""
    pass


class FeatureRejected(FeatureValidationError):
    """The candidate feature was rejected"""
    pass


class InvalidProjectStructure(FeatureValidationError):
    """An invalid change to the ballet project structure was introduced"""
    pass


class InvalidFeatureApi(FeatureValidationError):
    """The candidate feature had an invalid API"""
    pass


class FeatureCollectionError(BalletError):
    """Error in collecting Feature instances from some source"""
    pass


class NoFeaturesCollectedError(FeatureCollectionError):
    """Expected to collect some features but did not find any"""
    pass


class SkippedValidationTest(BalletError):
    """The validation test was skipped"""
    pass
