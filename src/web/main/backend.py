import json

__author__ = 'anushabala'
import uuid
from src.web.main.web_states import FinishedState, UserChatState, WaitingState
import hashlib
import sqlite3
import time
import datetime
import logging
import numpy as np
from src.basic.sessions.timed_session import TimedSessionWrapper
from src.basic.controller import Controller
from src.basic.event import Event
from src.basic.kb import KB
from flask import Markup
from uuid import uuid4


logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.FileHandler("chat.log")
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)
m = hashlib.md5()
m.update("bot")


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


class Partner(object):
    Human = 'human'
    Bot = m.hexdigest()


class Messages(object):
    ChatExpired = "Darn, you ran out of time! Waiting for a new chat..."
    PartnerConnectionTimeout = "Your partner's connection has timed out! Waiting for a new chat..."
    ConnectionTimeout = "Your connection has timed out! Waiting for a new chat..."
    YouLeftRoom = "You have left the room. Waiting for a new chat..."
    PartnerLeftRoom = "Your partner has left the room! Waiting for a new chat..."
    WaitingTimeExpired = "Sorry, no other users appear to be active at the moment. Please come back later!"
    ChatCompleted = "Great, you've completed the chat!"


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


class BackendConnection(object):
    def __init__(self, params, schema, scenario_db, systems, sessions, controller_map, pairing_probabilities, lexicon):
        self.config = params
        self.conn = sqlite3.connect(params["db"]["location"])
        self.lexicon = lexicon

        self.do_survey = True if "end_survey" in params.keys() and params["end_survey"] == 1 else False
        self.scenario_db = scenario_db
        self.schema = schema
        self.systems = systems
        self.sessions = sessions
        self.controller_map = controller_map
        self.pairing_probabilities = pairing_probabilities

    def _update_user(self, cursor, userid, **kwargs):
        if "status" in kwargs:
            logger.info("Updating status for user %s to %s" % (userid[:6], kwargs["status"]))
            kwargs["status_timestamp"] = current_timestamp_in_seconds()
        if "connected_status" in kwargs:
            logger.info("Updating connected status for user %s to %d" % (userid[:6], kwargs["connected_status"]))
            kwargs["connected_timestamp"] = current_timestamp_in_seconds()
        keys = sorted(kwargs.keys())
        values = [kwargs[k] for k in keys]
        set_string = ", ".join(["{}=?".format(k) for k in keys])

        cursor.execute("UPDATE active_user SET {} WHERE name=?".format(set_string), tuple(values + [userid]))

    def _get_session(self, userid):
        return self.sessions.get(userid)

    def _end_chat_and_transition_to_waiting(self, cursor, userid, partner_id, message, partner_message):
        logger.info("Removing users %s and %s from chat room - transition to WAIT" % (userid[:6], partner_id[:6]))

        controller = self.controller_map[userid]
        self.update_chat_reward(cursor, controller.get_chat_id(), controller.get_outcome())
        controller.set_inactive()
        self.controller_map[userid] = None
        self._update_user(cursor, userid,
                          status=Status.Waiting,
                          connected_status=1,
                          message=message)

    def _stop_waiting_and_transition_to_finished(self, cursor, userid):
        logger.info("User waiting duration exceeded time limit. Ending session for user.")
        self._update_user(cursor, userid,
                          status=Status.Finished,
                          message=Messages.WaitingTimeExpired)

    def _ensure_not_none(self, v, exception_class):
        if v is None:
            logger.warn("None: ", v)
            logger.warn("Raising exception %s" % type(exception_class).__name__)
            raise exception_class()
        else:
            return v

    @staticmethod
    def _is_timeout(timeout_limit, timestamp):
        if timeout_limit < 0:
            return False
        num_seconds_remaining = (timeout_limit + timestamp) - current_timestamp_in_seconds()
        return num_seconds_remaining <= 0

    def _assert_no_connection_timeout(self, connection_status, connection_timestamp):
        logger.debug("Checking for connection timeout: Connection status %d" % connection_status)
        if connection_status == 1:
            logger.debug("No connection timeout")
            return
        else:
            if self._is_timeout(self.config["connection_timeout_num_seconds"], connection_timestamp):
                raise ConnectionTimeoutException()
            else:
                logger.debug("No connection timeout")
                return

    def _assert_no_status_timeout(self, status, status_timestamp):
        N = self.config["status_params"][status]["num_seconds"]
        if N < 0:  # don't timeout for some statuses
            logger.debug("Checking for status timeout: no status timeout for status {}".format(status))
            return

        if self._is_timeout(N, status_timestamp):
            logger.warn("Checking for status timeout: Raising StatusTimeoutException")
            raise StatusTimeoutException()
        else:
            logger.debug("No status timeout")
            return

    @staticmethod
    def _validate_status_or_throw(assumed_status, status):
        logger.debug("Validating status: User status {}, assumed status {}".format(status, assumed_status))
        if status != assumed_status:
            logger.warn(
                "Validating status: User status {}, assumed status {} Raising UnexpectedStatusException".format(status,
                                                                                                                assumed_status))
            raise UnexpectedStatusException(status, assumed_status)
        return

    @staticmethod
    def _generate_chat_id():
        return "C_" + uuid4().hex

    def _get_user_info(self, cursor, userid, assumed_status=None):
        logger.debug("Getting info for user {} (assumed status: {})".format(userid[:6], assumed_status))
        u = self._get_user_info_unchecked(cursor, userid)
        if assumed_status is not None:
            self._validate_status_or_throw(assumed_status, u.status)
        self._assert_no_connection_timeout(u.connected_status, u.connected_timestamp)
        self._assert_no_status_timeout(u.status, u.status_timestamp)
        return u

    def _get_user_info_unchecked(self, cursor, userid):
        cursor.execute("SELECT * FROM active_user WHERE name=?", (userid,))
        x = cursor.fetchone()
        u = User(self._ensure_not_none(x, NoSuchUserException))
        return u

    def add_chat_to_db(self, chat_id, scenario_id):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('''INSERT INTO chat VALUES (?,?,"")''', (chat_id, scenario_id))

    def add_event_to_db(self, chat_id, event):
        def _create_row(chat_id,event):
            data = event.data
            if event.action == 'select':
                data = json.dumps(event.data)
            return chat_id, event.action, event.agent, event.time, data, event.start_time, json.dumps(event.metadata)
        try:
            with self.conn:
                cursor = self.conn.cursor()
                row = _create_row(chat_id, event)

                cursor.execute('''INSERT INTO event VALUES (?,?,?,?,?,?,?)''', row)
        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def attempt_join_chat(self, userid):
        def _init_controller(my_index, partner_type, scenario, chat_id):
            my_session = self.systems[Partner.Human].new_session(my_index, scenario.get_kb(my_index))
            partner_session = TimedSessionWrapper(1-my_index, self.systems[partner_type].new_session(1-my_index, scenario.get_kb(1-my_index), scenario.uuid))

            controller = Controller(scenario, [my_session, partner_session], chat_id=chat_id, debug=False)

            return controller, my_session, partner_session

        def _pair_with_human(cursor, userid, my_index, partner_id, scenario, chat_id):
            controller, my_session, partner_session = _init_controller(my_index, Partner.Human, scenario, chat_id)
            self.controller_map[userid] = controller
            self.controller_map[partner_id] = controller

            self.sessions[userid] = my_session
            self.sessions[partner_id] = partner_session

            self._update_user(cursor, partner_id,
                              status=Status.Chat,
                              partner_id=userid,
                              scenario_id=scenario.uuid,
                              agent_index=1 - my_index,
                              selected_index=-1,
                              message="",
                              chat_id=chat_id)

            self._update_user(cursor, userid,
                              status=Status.Chat,
                              partner_id=partner_id,
                              scenario_id=scenario_id,
                              agent_index=my_index,
                              selected_index=-1,
                              message="",
                              chat_id=chat_id)

            return True

        def _pair_with_bot(cursor, userid, my_index, bot_type, scenario, chat_id):
            controller, my_session, bot_session = _init_controller(my_index, bot_type, scenario, chat_id)
            self.controller_map[userid] = controller

            self.sessions[userid] = my_session

            self._update_user(cursor, userid,
                              status=Status.Chat,
                              partner_id=Partner.Bot,
                              scenario_id=scenario_id,
                              agent_index=my_index,
                              selected_index=-1,
                              message="",
                              chat_id=chat_id)

            return True

        def _get_other_waiting_users(cursor, userid):
            cursor.execute("SELECT name FROM active_user WHERE name!=? AND status=? AND connected_status=1",
                           (userid, Status.Waiting))
            userids = [r[0] for r in cursor.fetchall()]
            return userids

        def _choose_new_scenario(cursor):
            cursor.execute("SELECT scenario_id FROM chat")
            prev_scenarios = set(r[0] for r in cursor.fetchall())
            return self.scenario_db.select_random(prev_scenarios)

        try:
            with self.conn:
                cursor = self.conn.cursor()
                others = _get_other_waiting_users(cursor, userid)

                partner_types = self.pairing_probabilities.keys()
                partner_probs = self.pairing_probabilities.values()

                partner_type = np.random.choice(partner_types, p=partner_probs)

                my_index = np.random.choice([0, 1])
                scenario = _choose_new_scenario(cursor)
                scenario_id = scenario.uuid
                chat_id = self._generate_chat_id()
                if partner_type == Partner.Human:
                    if len(others) == 0:
                        return None
                    self.add_chat_to_db(chat_id, scenario_id)
                    partner_id = np.random.choice(others)
                    return _pair_with_human(cursor, userid, my_index, partner_id, scenario, chat_id)
                else:
                    self.add_chat_to_db(chat_id, scenario_id)
                    return _pair_with_bot(cursor, userid, my_index, partner_type, scenario, chat_id)

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def check_game_over_and_transition(self, cursor, userid, partner_id):
        def _user_finished(cursor, userid):
            new_status = Status.Survey if self.do_survey else Status.Finished
            message = Messages.ChatCompleted
            self._update_user(cursor, userid,
                              status=new_status,
                              message=message,
                              partner_id=-1)

        if self.is_game_over(userid):
            controller = self.controller_map[userid]
            controller.set_inactive()
            self.update_chat_reward(cursor, controller.get_chat_id(), controller.get_outcome())
            self.controller_map[userid] = None
            _user_finished(cursor, userid)

            if not self.is_user_partner_bot(cursor, userid):
                _user_finished(cursor, partner_id)
            return True

        return False

    def close(self):
        self.conn.close()
        self.conn = None

    def connect(self, userid):
        try:
            with self.conn:
                cursor = self.conn.cursor()
                self._update_user(cursor, userid,
                                  connected_status=1)
        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def create_user_if_not_exists(self, username):
        with self.conn:
            cursor = self.conn.cursor()
            now = current_timestamp_in_seconds()
            cursor.execute('''INSERT OR IGNORE INTO active_user VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
                           (username, Status.Waiting, now, 0, now, "", "", "", "", -1, -1, ""))

    def disconnect(self, userid):
        try:
            with self.conn:
                cursor = self.conn.cursor()
                self._update_user(cursor, userid,
                                  connected_status=0)

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def get_chat_info(self, userid):
        try:
            with self.conn:
                logger.info("Getting chat info for user %s" % userid[:6])
                cursor = self.conn.cursor()
                u = self._get_user_info_unchecked(cursor, userid)
                num_seconds_remaining = (self.config["status_params"]["chat"]["num_seconds"] +
                                         u.status_timestamp) - current_timestamp_in_seconds()
                scenario = self.scenario_db.get(u.scenario_id)
                return UserChatState(u.agent_index, scenario.uuid, u.chat_id, scenario.get_kb(u.agent_index),
                                     scenario.attributes, num_seconds_remaining)

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def get_finished_info(self, userid, from_mturk=False, completed=True):
        def _generate_mturk_code(completed=True):
            if completed:
                return "MTURK_TASK_C{}".format(str(uuid.uuid4().hex))
            return "MTURK_TASK_I{}".format(str(uuid.uuid4().hex))

        def _add_finished_task_row(cursor, userid, mturk_code, chat_id):
            logger.info(
                "Adding row into mturk_task: userid={},mturk_code={}".format(
                    userid[:6],
                    mturk_code))
            cursor.execute('INSERT INTO mturk_task VALUES (?,?,?)',
                           (userid, mturk_code, chat_id))

        try:
            logger.info("Trying to get finished session info for user %s" % userid[:6])
            with self.conn:
                cursor = self.conn.cursor()
                u = self._get_user_info(cursor, userid, assumed_status=Status.Finished)
                num_seconds = (self.config["status_params"]["finished"][
                                   "num_seconds"] + u.status_timestamp) - current_timestamp_in_seconds()
                if from_mturk:
                    logger.info("Generating mechanical turk code for user %s" % userid[:6])
                    mturk_code = _generate_mturk_code(completed)
                else:
                    mturk_code = None
                _add_finished_task_row(cursor, userid, mturk_code, u.chat_id)
                return FinishedState(Markup(u.message), num_seconds, mturk_code)

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def get_waiting_info(self, userid):
        try:
            with self.conn:
                logger.info("Getting waiting session info for user %s" % userid[:6])
                cursor = self.conn.cursor()
                u = self._get_user_info(cursor, userid, assumed_status=Status.Waiting)
                num_seconds = (self.config["status_params"]["waiting"]["num_seconds"] +
                               u.status_timestamp) - current_timestamp_in_seconds()
                return WaitingState(u.message, num_seconds)

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def get_schema(self):
        return self.schema

    def get_updated_status(self, userid):
        try:
            logger.debug("Getting current status for user %s" % userid[:6])
            with self.conn:
                cursor = self.conn.cursor()
                try:
                    u = self._get_user_info(cursor, userid, assumed_status=None)
                    logger.debug("Got user info for user %s without exceptions. Returning status %s" % (
                        userid[:6], u.status))
                    return u.status
                except (UnexpectedStatusException, ConnectionTimeoutException, StatusTimeoutException) as e:
                    logger.warn("Caught %s while getting status for user %s" % (type(e).__name__, userid[:6]))
                    if isinstance(e, UnexpectedStatusException):
                        logger.warn(
                            "Unexpected behavior: got UnexpectedStatusException while getting user status")  # this should never happen
                    # Handle timeouts by performing the relevant update
                    u = self._get_user_info_unchecked(cursor, userid)
                    logger.debug("Unchecked user status for user %s: %s" % (userid[:6], u.status))

                    if u.status == Status.Waiting:
                        if isinstance(e, ConnectionTimeoutException):
                            logger.info(
                                "User %s had connection timeout in waiting state. Updating connection status to "
                                "connected to reenter waiting state." % userid[:6])
                            self._update_user(cursor, userid, connected_status=1, status=Status.Waiting)
                            return u.status
                        else:
                            logger.info("User %s had status timeout in waiting state." % userid[:6])
                            self._stop_waiting_and_transition_to_finished(cursor, userid)
                            return Status.Finished

                    elif u.status == Status.Chat:
                        if isinstance(e, ConnectionTimeoutException):
                            logger.info(
                                "User %s had connection timeout in chat state. Updating connection status to connected "
                                "and reentering waiting state." % userid[:6])
                            message = Messages.PartnerConnectionTimeout
                        else:
                            logger.info(
                                "Chat timed out for user %s. Leaving chat room and entering waiting state."
                                % userid[:6])
                            message = Messages.ChatExpired

                        self._end_chat_and_transition_to_waiting(cursor, userid, u.partner_id, message=message,
                                                                 partner_message=message)
                        return Status.Waiting

                    elif u.status == Status.Finished:
                        logger.info(
                            "User %s was previously in finished state. Updating to waiting state with connection "
                            "status = connected." % userid[:6])
                        self._update_user(cursor, userid, connected_status=1, status=Status.Waiting, message="")
                        return Status.Waiting

                    elif u.status == Status.Survey:
                        if isinstance(e, ConnectionTimeoutException):
                            logger.info('ConnectionTimeOutException for user %s in survey state. Updating to connected '
                                        'and reentering waiting state.' % userid[:6])
                            self._update_user(cursor, userid, connected_status=1, status=Status.Waiting)
                            return Status.Waiting
                        else:
                            return Status.Survey # this should never happen because surveys can't time out
                    else:
                        raise Exception("Unknown status: {} for user: {}".format(u.status, userid))

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def get_user_message(self, userid):
        with self.conn:
            cursor = self.conn.cursor()
            u = self._get_user_info_unchecked(cursor, userid)
            return u.message

    def is_chat_valid(self, userid):
        try:
            with self.conn:
                cursor = self.conn.cursor()
                try:
                    u = self._get_user_info(cursor, userid, assumed_status=Status.Chat)
                except UnexpectedStatusException:
                    return False
                except StatusTimeoutException:
                    u = self._get_user_info_unchecked(cursor, userid)
                    logger.debug("User {} had status timeout.".format(u.name[:6]))
                    self._end_chat_and_transition_to_waiting(cursor, userid, u.partner_id, message=Messages.YouLeftRoom,
                                                             partner_message=Messages.PartnerLeftRoom)
                    return False
                except ConnectionTimeoutException:
                    return False

                if not self.is_user_partner_bot(cursor, userid):
                    try:
                        u2 = self._get_user_info(cursor, u.partner_id, assumed_status=Status.Chat)
                    except UnexpectedStatusException:
                        self._end_chat_and_transition_to_waiting(cursor, userid, None, message=Messages.PartnerLeftRoom,
                                                                 partner_message=None)
                        return False
                    except StatusTimeoutException:
                        self._end_chat_and_transition_to_waiting(cursor, userid, u.partner_id, message=Messages.ChatExpired,
                                                                 partner_message=Messages.ChatExpired)
                        return False
                    except ConnectionTimeoutException:
                        self._end_chat_and_transition_to_waiting(cursor, userid, u.partner_id,
                                                                 message=Messages.PartnerLeftRoom,
                                                                 partner_message=Messages.YouLeftRoom)
                        return False

                    if self.controller_map[userid] != self.controller_map[u.partner_id]:
                        return False

                if self.check_game_over_and_transition(cursor, userid, u.partner_id):
                    return False

                return True

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")
            return False

    def is_connected(self, userid):
        try:
            with self.conn:
                cursor = self.conn.cursor()
                try:
                    u = self._get_user_info(cursor, userid)
                    return True if u.connected_status == 1 else False
                except (UnexpectedStatusException, ConnectionTimeoutException, StatusTimeoutException) as e:
                    return False

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def is_game_over(self, userid):
        controller = self.controller_map[userid]
        return controller.game_over()

    def is_user_partner_bot(self, cursor, userid):
        u = self._get_user_info_unchecked(cursor, userid)
        return u.partner_id is not None and u.partner_id == Partner.Bot

    def is_status_unchanged(self, userid, assumed_status):
        try:
            with self.conn:
                cursor = self.conn.cursor()
                try:
                    logger.debug("Checking whether status has changed from %s for user %s" % (
                        assumed_status, userid[:6]))
                    u = self._get_user_info(cursor, userid, assumed_status=assumed_status)
                    if u.status == Status.Waiting:
                        logger.debug("User %s is waiting. Checking if other users are available for chat..")
                        self.attempt_join_chat(userid)
                    logger.debug("Returning TRUE (user status hasn't changed)")
                    return True
                except (UnexpectedStatusException, ConnectionTimeoutException, StatusTimeoutException) as e:
                    logger.warn(
                        "Caught %s while getting status for user %s. Returning FALSE (user status has changed)" % (
                            type(e).__name__, userid[:6]))
                    if isinstance(e, UnexpectedStatusException):
                        logger.warn("Found status %s, expected (assumed) status %s" % (
                            e.found_status, e.expected_status))
                    return False
        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def receive(self, userid):
        controller = self.controller_map[userid]
        if controller is None:
            # fail silently - this just means that receive is called between the time that the chat has ended and the
            # time that the page is refreshed
            return None
        controller.step(self)
        session = self._get_session(userid)
        return session.poll_inbox()

    def select(self, userid, idx):
        try:
            with self.conn:
                cursor = self.conn.cursor()
                u = self._get_user_info_unchecked(cursor, userid)
                scenario = self.scenario_db.get(u.scenario_id)
                kb = scenario.get_kb(u.agent_index)
                item = kb.items[idx]
                self.send(userid, Event.SelectionEvent(u.agent_index,
                                                       item,
                                                       str(time.time())))
                return item
        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")
            return None

    def send(self, userid, event):
        session = self._get_session(userid)
        session.enqueue(event)
        controller = self.controller_map[userid]
        if controller is None:
            # fail silently because this just means that the user tries to send something after their partner has left
            # (but before the chat has ended)
            return None
        controller.step(self)
        # self.add_event_to_db(controller.get_chat_id(), event)

    def submit_survey(self, userid, data):
        def _user_finished(userid):
            logger.info("Updating user %s to status FINISHED from status survey" % userid)
            self._update_user(cursor, userid, status=Status.Finished)

        try:
            with self.conn:
                cursor = self.conn.cursor()
                user_info = self._get_user_info_unchecked(cursor, userid)

                cursor.execute('INSERT INTO survey VALUES (?,?,?,?,?)',
                               (userid, user_info.chat_id, user_info.partner_type, data['question1'], data['question2']))
                _user_finished(userid)
        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def update_chat_reward(self, cursor, chat_id, outcome):
        str_outcome = json.dumps(outcome)
        cursor.execute('''UPDATE chat SET outcome=? WHERE chat_id=?''', (str_outcome, chat_id))
