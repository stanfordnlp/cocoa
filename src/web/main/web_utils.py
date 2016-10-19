__author__ = 'anushabala'
from flask import g
from flask import current_app as app

from backend import BackendConnection

DATE_FMT = '%m-%d-%Y:%H-%M-%S'


def get_backend():
    backend = getattr(g, '_backend', None)
    if backend is None:
        backend = g._backend = BackendConnection(app.config["user_params"],
                                                 app.config["schema"],
                                                 app.config["scenario_db"],
                                                 app.config["systems"],
                                                 app.config["sessions"],
                                                 app.config["controller_map"],
                                                 app.config["pairing_probabilities"],
                                                 app.config["lexicon"])
    return backend