__author__ = 'anushabala'
from flask import g
from flask import current_app as app
import time
import datetime

from backend import BackendConnection


def get_backend():
    backend = getattr(g, '_backend', None)
    if backend is None:
        backend = g._backend = BackendConnection.get_backend(app.config["user_params"],
                                                 app.config["schema"],
                                                 app.config["scenario_db"],
                                                 app.config["systems"],
                                                 app.config["sessions"],
                                                 app.config["controller_map"],
                                                 app.config["pairing_probabilities"],
                                                 app.config["lexicon"])
    return backend