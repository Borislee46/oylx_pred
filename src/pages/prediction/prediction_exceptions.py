class PredictionError(Exception):
    pass


class ModelLoadError(PredictionError):
    pass


class InvalidInputError(PredictionError):
    pass


class PredictionExecutionError(PredictionError):
    pass


class DataValidationError(PredictionError):
    pass

