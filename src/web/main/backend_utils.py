__author__ = 'anushabala'
import time
import datetime
import src.config as config


class Status(object):
    Waiting = "waiting"
    Chat = "chat"
    Finished = "finished"
    Survey = "survey"
    Redirected = "redirected"


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
    Waiting = 'Waiting for a new chat...'

    NegotiationCompleted = "Great, you reached a final offer!"
    NegotiationIncomplete = "Sorry, you weren't able to reach a deal. :("
    NegotiationBetterDeal = "Congratulations, you got the better deal! We'll award you a bonus on Mechanical Turk."
    NegotiationWorseDeal = "Sorry, your partner got the better deal. :("
    NegotiationRedirect = "Sorry, that chat did not meet our acceptance criteria."


    @staticmethod
    def get_completed_message():
        if config.task == config.MutualFriends:
            return Messages.ChatCompleted
        elif config.task == config.Negotiation:
            return Messages.NegotiationCompleted

    @staticmethod
    def get_incomplete_message():
        if config.task == config.MutualFriends:
            # todo this shouldn't really matter because there isn't a way to "quit" tasks in the MF world
            return Messages.ConnectionTimeout
        elif config.task == config.Negotiation:
            return Messages.NegotiationIncomplete


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