from flask import Blueprint, request, session, redirect, url_for, render_template

from src.models.users.user import User

from src.models.portfolios.portfolio import Portfolio
import src.models.users.errors as UserErrors
import src.models.users.decorators as user_decorators
import datetime
user_blueprint = Blueprint('users', __name__)

@user_blueprint.route('/login', methods = ['GET','POST'])
def login_user():   # Views form required for user login
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            if User.is_login_valid(email,password):
                session['email'] = email
                ## CHANGE ##
                return redirect(url_for(".user_profile"))
        except UserErrors.UserError as e:
            return e.message

    return render_template("users/login.jinja2")

@user_blueprint.route('/register', methods = ['GET','POST'])
def register_user():   # Views form required for user signup
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        try:
            if User.register_user(email,password, first_name, last_name):
                session['email'] = email
                return redirect(url_for(".user_profile"))
        except UserErrors.UserError as e:
            return e.message

    return render_template("users/register.jinja2")

@user_blueprint.route('/logout')
def logout_user():  # Logs user out
    session['email'] = None
    return redirect(url_for('home'))

@user_blueprint.route('/portfolios')
@user_decorators.requires_login
def user_portfolios():  # Views list of user portfolios
    user = User.find_by_email(session['email'])
    portfolios = user.get_portfolios()

    return render_template('/users/portfolios.jinja2', portfolios = portfolios, user = user)
@user_blueprint.route('/update_profile', methods = ['GET','POST'])
@user_decorators.requires_login
def update_profile():  # Views list of user portfolios
    if request.method == 'POST':
        age = request.form['age']
        position = request.form['position']
        current_asset = request.form['current_asset']
        current_debt = request.form['current_debt']
        user = User.find_by_email(session['email'])
        user.update_profile(age, position,float(current_asset),float(current_debt))
        return redirect(url_for(".user_profile"))

    return render_template("users/update_profile.jinja2")


@user_blueprint.route('/profile')
@user_decorators.requires_login
def user_profile():  # Views list of user portfolios
    user = User.find_by_email(session['email'])
    user.update_balance()
    ports= Portfolio.get_by_email(session['email'])
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template('/users/profile_page.jinja2', user = user, ports=ports, current_date = date)
