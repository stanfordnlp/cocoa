__author__ = 'anushabala'
import time
import datetime
#import src.config as config


class Status(object):
    Waiting = "waiting"
    Chat = "chat"
    Finished = "finished"
    Survey = "survey"
    Redirected = "redirected"
    Incomplete = "incomplete"
    Reporting = "reporting"


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
    ConnectionTimeout = "Your connection has timed out. Please reenter this website using the original URL provided to " \
                        "you to start a new chat."
    YouLeftRoom = 'You skipped the chat. '
    PartnerLeftRoom = 'Your partner has left the chat!'
    WaitingTimeExpired = "Sorry, no other users appear to be active at the moment. Please come back later!"
    ChatCompleted = "Great, you've completed the chat!"
    ChatIncomplete = ConnectionTimeout
    HITCompletionWarning = "Please note that you will only get credit for this HIT if you made a good attempt to complete the chat."
    Waiting = 'Waiting for a new chat...'


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
