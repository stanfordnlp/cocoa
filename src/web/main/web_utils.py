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

class Status(object):
    Waiting = "waiting"
    Chat = "chat"
    Finished = "finished"
    Survey = "survey"


class UnexpectedStatusException(Exception):
    def __init__(self, found_status, expected_status):
        self.expected_status = expected_status
        self.found_status = found_status


class ConnectionTimeoutException(Exception):
    pass


class InvalidStatusException(Exception):
    pass


class StatusTimeoutException(Exception):
    pass


class NoSuchUserException(Exception):
    pass


class Messages(object):
    ChatExpired = 'You ran out of time!'
    PartnerConnectionTimeout = "Your partner's connection has timed out! Waiting for a new chat..."
    ConnectionTimeout = "Your connection has timed out!"
    YouLeftRoom = 'You skipped the chat. '
    PartnerLeftRoom = 'Your partner has left the chat!'
    WaitingTimeExpired = "Sorry, no other users appear to be active at the moment. Please come back later!"
    ChatCompleted = "Great, you've completed the chat!"
    HITCompletionWarning = "Please note that you will only get credit for this HIT if you made a good attempt to complete the chat."


def current_timestamp_in_seconds():
    return int(time.mktime(datetime.datetime.now().timetuple()))


class User(object):
    def __init__(self, row):
        self.name = row[0]
        self.status = row[1]
        self.status_timestamp = row[2]
        self.connected_status = row[3]
        self.connected_timestamp = row[4]
        self.message = row[5]
        self.partner_type = row[6]
        self.partner_id = row[7]
        self.scenario_id = row[8]
        self.agent_index = row[9]
        self.selected_index = row[10]
        self.chat_id = row[11]