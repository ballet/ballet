class Error(Exception):
    """Base error for ballet"""
    pass


class UnexpectedTravisEnvironmentError(Error):
    """The environment within Travis CI testing was unexpected"""
    pass


class UnsuccessfulInputConversionError(Error):
    """Input-type conversion for execution within pipeline was unsuccessful"""
    pass


class ConfigurationError(Error):
    """Error in configuration of the ballet project"""
    pass


class FeatureValidationError(Error):
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


class SkippedValidationTest(Error):
    """The validation test was skipped"""
    pass
