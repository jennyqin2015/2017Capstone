from flask import Blueprint, request, session, redirect, url_for, render_template

from src.models.users.user import User
import src.models.users.errors as UserErrors
import src.models.users.decorators as user_decorators
from src.common.database import Database
from src.models.portfolios.portfolio import Portfolio

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import urllib
import base64


portfolio_blueprint = Blueprint('portfolios', __name__)

@portfolio_blueprint.route('/portfolio/<string:portfolio_id>')
@user_decorators.requires_login
def get_portfolio_page(portfolio_id):   # Renders unique portfolio page
    port = Portfolio.get_by_id(portfolio_id)
    fig = port.plot_portfolio()
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    plot_data = base64.b64encode(img.read()).decode()

    return render_template('/portfolios/portfolio.jinja2', portfolio = port, plot_url=plot_data)

@portfolio_blueprint.route('/edit/<string:portfolio_id>', methods=['GET','POST'])
@user_decorators.requires_login
def change_risk(portfolio_id):         # Views form to change portfolio's associated risk aversion parameter
    port = Portfolio.get_by_id(portfolio_id)
    if request.method == "POST":
        risk_appetite = request.form['risk_appetite']
        port.risk_appetite = risk_appetite
        port.save_to_mongo()
        fig = port.runMVO()
        canvas = FigureCanvas(fig)
        img = BytesIO()
        fig.savefig(img)
        img.seek(0)
        plot_data = base64.b64encode(img.read()).decode()
        return render_template('/portfolios/optimal_portfolio.jinja2', portfolio = port, plot_url=plot_data)

    return render_template('/portfolios/edit_portfolio.jinja2',portfolio = port)

@portfolio_blueprint.route('/new', methods=['GET','POST'])
@user_decorators.requires_login
def create_portfolio():            # Views form to create portfolio associated with active/ loggedin user
    if request.method == "POST":
        risk_appetite = request.form['risk_appetite']
        port = Portfolio(session['email'], risk_appetite= risk_appetite)
        port.save_to_mongo()
        fig = port.runMVO()
        canvas = FigureCanvas(fig)
        img = BytesIO()
        fig.savefig(img)
        img.seek(0)
        plot_data = base64.b64encode(img.read()).decode()
        return render_template('/portfolios/optimal_portfolio.jinja2', portfolio=port, plot_url=plot_data)

    return render_template('/portfolios/new_portfolio.jinja2')
