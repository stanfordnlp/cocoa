__author__ = 'anushabala'

from flask import Blueprint

main = Blueprint('main', __name__)
from . import routes, web_utils, backend