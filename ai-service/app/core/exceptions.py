class AppException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class RetrievalException(AppException):
    pass


class GenerationException(AppException):
    pass


class ValidationException(AppException):
    pass