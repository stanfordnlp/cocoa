__author__ = 'anushabala'

from flask import Blueprint

main = Blueprint('main', __name__)
from . import web_utils, backend
from routing import routes