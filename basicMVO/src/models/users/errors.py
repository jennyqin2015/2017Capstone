class UserError(Exception):
    def __init__(self, message):
        self.message = message

class UserNotExistsError(UserError):    # Raises an error if user is not stored in MongoDB
    pass

class IncorrectPasswordError(UserError):  # Raises an error if password does not match the one stored in MongoDB
    pass

class UserAlreadyRegisteredError(UserError):    # Raises an error if user tries to signup even though already available in MongoDB
    pass

class InvalidEmailError(UserError): # Raises an error if email entered does not follow an acceptable email regex pattern
    pass
