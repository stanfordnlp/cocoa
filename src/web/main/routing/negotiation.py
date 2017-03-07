__author__ = 'anushabala'

import uuid
import logging
from datetime import datetime
import time

from flask import jsonify, render_template, request, redirect, url_for, Markup
from flask import current_app as app

from .. import main
from src.web.main.web_utils import get_backend
from src.web.main.backend import Status
from src.basic.event import Event
from routes import generate_userid, userid, userid_prefix, generate_unique_key, format_message




