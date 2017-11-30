import uuid

from src.common.database import Database
import src.models.users.errors as UserErrors
from src.common.util import Utils
import src.models.users.constants as UserConstants
import src.models.portfolios.constants as PortfolioConstants
from src.models.portfolios.portfolio import Portfolio


class User(object):
    def __init__(self, email, password, first_name, last_name, _id = None):
        self.email = email
        self.password = password
        self.first_name = first_name
        self.last_name = last_name
        self._id = uuid.uuid4().hex if _id is None else _id

    def __repr__(self):
        return "<User {}>".format(self.email)

    @staticmethod
    def is_login_valid(email, password):
        '''
        This method verifies that an email-password combo (as sent by site forms) is valid or not.
        Checks that email exists, and that password associated to that email is correct

        :param email: The user's email
        :param password: A sha512 hashed password
        :return: True if valid, False otherwise
        '''
        user_data = Database.find_one(UserConstants.COLLECTION,
                                      {'email': email})  # Password in sha512 --> pbkdf2_sha512

        if user_data is None:
            # Tell the user their email does not exist
            raise UserErrors.UserNotExistsError("Your user does not exist!")
        if not Utils.check_hashed_password(password, user_data['password']):
            # Tell the user their password is wrong
            raise UserErrors.IncorrectPasswordError("Your password was wrong!")

        return True

    @staticmethod
    def register_user(email, password, first_name, last_name):
        '''
        Registers a user using an email & password. Password already comes hashed as sha-512.

        :param email: user's email (might be invalid)
        :param password: sha-512 hashed password
        :return: True if registered, False otherwise (exceptions can als be raised)
        '''

        user_data = Database.find_one(UserConstants.COLLECTION, {"email": email})

        if user_data is not None:
            raise UserErrors.UserAlreadyRegisteredError("User email already exists")

        if not Utils.email_is_valid(email):
            raise UserErrors.InvalidEmailError("Invalid email format!")

        User(email, Utils.hash_password(password), first_name, last_name).save_to_mongo()
        return True

    def save_to_mongo(self):
        Database.insert(UserConstants.COLLECTION, data=self.json())

    def json(self):      # Creates JSON representation of user instance
        return {
            "_id": self._id,
            "email": self.email,
            "password": self.password,
            "first_name": self.first_name,
            "last_name": self.last_name
        }

    @classmethod
    def find_by_email(cls, email):  # Retrieves user record by unique email
        return cls(**Database.find_one(UserConstants.COLLECTION, {'email': email}))

    def get_portfolios(self):  # Retrieves portfolio(s) associated with user by unique email
        return Portfolio.get_by_email(self.email)
