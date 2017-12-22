import uuid

from src.common.database import Database
import src.models.users.errors as UserErrors
from src.common.util import Utils
import src.models.users.constants as UserConstants
import src.models.portfolios.constants as PortfolioConstants
from src.models.portfolios.portfolio import Portfolio


class User(object):
    def __init__(self, email, password, first_name, last_name, age = None, position = None, current_asset =None, current_debt = None, number_goals = None, total_balance = None, _id = None):
        self.email = email
        self.password = password
        self.first_name = first_name
        self.last_name = last_name
        self.age = 0 if age is None else age
        self.position = "Unknown" if position is None else position
        self.current_asset = 0 if current_asset is None else current_asset
        self.current_debt = 0 if current_debt is None else current_debt
        self.number_goals = 0 if number_goals is None else number_goals
        self.total_balance = 0 if total_balance is None else total_balance
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


    def update_profile(self, age, position, current_asset, current_debt):
        self.age = age
        self.position = position
        self.current_asset = current_asset
        self.current_debt = current_debt
        self.save_to_mongo()
        return
    def update_balance(self):
        portfolios = Portfolio.get_by_email(self.email)
        total_balance=0
        for i in portfolios:
            total_balance += i.account_balance[-1]
        self.total_balance = total_balance
        self.save_to_mongo()
        return
    def save_to_mongo(self):
        Database.update(UserConstants.COLLECTION,{'email': self.email}, self.json())

    def json(self):      # Creates JSON representation of user instance
        return {
            "_id": self._id,
            "email": self.email,
            "password": self.password,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "age": self.age,
            "position": self.position,
            "current_asset": self.current_asset,
            "current_debt": self.current_debt,
            "number_goals": self.number_goals,
            "total_balance": self.total_balance
        }

    @classmethod
    def find_by_email(cls, email):  # Retrieves user record by unique email
        return cls(**Database.find_one(UserConstants.COLLECTION, {'email': email}))

    def get_portfolios(self):  # Retrieves portfolio(s) associated with user by unique email
        return Portfolio.get_by_email(self.email)
