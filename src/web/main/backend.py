__author__ = 'anushabala'
import uuid
from src.web.main.web_sessions import FinishedSession, UserChatSession, WaitingSession
import hashlib
import sqlite3
import time
import datetime
import logging
import numpy as np
from src.basic.controller import Controller
from src.basic.event import Event
from flask import Markup
import os


date_fmt = '%m-%d-%Y:%H-%M-%S'
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.FileHandler("chat.log")
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)
m = hashlib.md5()
m.update("bot")


class Status(object):
    Waiting, Chat, SingleTask, Finished, Survey = range(5)
    _names = ["waiting", "chat", "single_task", "finished", "survey"]

    @staticmethod
    def from_str(s):
        if Status._names.index(s) == 0:
            return Status.Waiting
        if Status._names.index(s) == 1:
            return Status.Chat
        if Status._names.index(s) == 2:
            return Status.SingleTask
        if Status._names.index(s) == 3:
            return Status.Finished
        if Status._names.index(s) == 4:
            return Status.Survey


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
    PartnerConnectionTimeout = "Your friend's connection has timed out! Waiting for a new chat..."
    ConnectionTimeout = "Your connection has timed out! Waiting for a new chat..."
    YouLeftRoom = "You have left the room. Waiting for a new chat..."
    PartnerLeftRoom = "Your friend has left the room! Waiting for a new chat..."
    WaitingTimeExpired = "Sorry, no other users appear to be active at the moment."
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
        self.room_id = row[6]
        self.partner_type = row[7]
        self.partner_id = row[8]
        self.scenario_id = row[9]
        self.agent_index = row[10]
        self.selected_index = row[11]
        self.num_chats_completed = row[12]


class BackendConnection(object):
    def __init__(self, params, schema, scenario_db, systems, sessions, controller_map, controller_queue,
                 pairing_probabilities, lexicon):
        self.config = params
        self.conn = sqlite3.connect(params["db"]["location"])
        self.lexicon = lexicon

        self.do_survey = True if "end_survey" in params.keys() and params["end_survey"] == 1 else False
        self.scenario_db = scenario_db
        self.schema = schema
        self.systems = systems
        self.sessions = sessions
        self.controller_map = controller_map
        self.controller_queue = controller_queue
        self.pairing_probabilities = pairing_probabilities

    def _update_user(self, cursor, userid, **kwargs):
        if "status" in kwargs:
            logger.info("Updating status for user %s to %s" % (userid[:6], Status._names[kwargs["status"]]))
            kwargs["status_timestamp"] = current_timestamp_in_seconds()
        if "connected_status" in kwargs:
            logger.info("Updating connected status for user %s to %d" % (userid[:6], kwargs["connected_status"]))
            kwargs["connected_timestamp"] = current_timestamp_in_seconds()
        keys = sorted(kwargs.keys())
        values = [kwargs[k] for k in keys]
        set_string = ", ".join(["{}=?".format(k) for k in keys])

        cursor.execute("UPDATE ActiveUsers SET {} WHERE name=?".format(set_string), tuple(values + [userid]))

    def _get_session(self, userid):
        return self.sessions.get(userid)

    def _end_chat_and_transition_to_waiting(self, cursor, userid, partner_id, message, partner_message):
        logger.info("Removing users %s and %s from chat room - transition to WAIT" % (userid[:6], partner_id[:6]))

        controller = self.controller_map[userid]
        controller.set_inactive()
        self.dump_transcript(cursor, controller, userid)
        self.controller_map[userid] = None
        self._update_user(cursor, userid,
                          status=Status.Waiting,
                          room_id=-1,
                          connected_status=1,
                          message=message)
        if partner_id != Partner.Bot:
            self._update_user(cursor, partner_id,
                              status=Status.Waiting,
                              room_id=-1,
                              message=partner_message)

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

    def _assert_no_connection_timeout(self, connection_status, connection_timestamp):
        logger.debug("Checking for connection timeout: Connection status %d" % connection_status)
        if connection_status == 1:
            logger.debug("No connection timeout")
            return
        else:
            N = self.config["connection_timeout_num_seconds"]
            num_seconds_remaining = (N + connection_timestamp) - current_timestamp_in_seconds()
            if num_seconds_remaining >= 0:
                logger.debug("Timeout limit: %d Status timestamp: %d Seconds remaining: %d" % (
                    N, connection_timestamp, num_seconds_remaining))
                logger.debug("No connection timeout")
                return
            else:
                logger.info("Timeout limit: %d Status timestamp: %d Seconds remaining: %d" % (
                    N, connection_timestamp, num_seconds_remaining))
                logger.warn("Checking for connection timeout: Raising ConnectionTimeoutException")
                raise ConnectionTimeoutException()

    def _assert_no_status_timeout(self, status, status_timestamp):
        N = self.config["status_params"][Status._names[status]]["num_seconds"]
        if N < 0:  # don't timeout for some statuses
            logger.debug("Checking for status timeout: no status timeout for status {}".format(Status._names[status]))
            return
        num_seconds_remaining = (N + status_timestamp) - current_timestamp_in_seconds()

        if num_seconds_remaining >= 0:
            logger.debug("No status timeout")
            logger.debug(
                "Checking for timeout of status '%s': Seconds for status: %d Status timestamp: %d Seconds remaining: %d" % (
                    Status._names[status], N, status_timestamp, num_seconds_remaining))
            return
        else:
            logger.info(
                "Checking for timeout of status '%s': Seconds for status: %d Status timestamp: %d Seconds remaining: %d" % (
                    Status._names[status], N, status_timestamp, num_seconds_remaining))
            logger.warn("Checking for status timeout: Raising StatusTimeoutException")
            raise StatusTimeoutException()

    @staticmethod
    def _validate_status_or_throw(assumed_status, status):
        logger.debug("Validating status: User status {}, assumed status {}".format(status, assumed_status))
        if status != assumed_status:
            logger.warn(
                "Validating status: User status {}, assumed status {} Raising UnexpectedStatusException".format(status,
                                                                                                                assumed_status))
            raise UnexpectedStatusException(status, assumed_status)
        return

    def _get_user_info(self, cursor, userid, assumed_status=None):
        logger.debug("Getting info for user {} (assumed status: {})".format(userid[:6], assumed_status))
        u = self._get_user_info_unchecked(cursor, userid)
        if assumed_status is not None:
            self._validate_status_or_throw(assumed_status, u.status)
        self._assert_no_connection_timeout(u.connected_status, u.connected_timestamp)
        self._assert_no_status_timeout(u.status, u.status_timestamp)
        return u

    def _get_user_info_unchecked(self, cursor, userid):
        cursor.execute("SELECT * FROM ActiveUsers WHERE name=?", (userid,))
        x = cursor.fetchone()
        u = User(self._ensure_not_none(x, NoSuchUserException))
        return u

    def attempt_join_room(self, userid):
        def _init_controller(my_index, partner_type, scenario):
            my_session = self.systems[Partner.Human].new_session(my_index, scenario.get_kb(my_index))
            partner_session = self.systems[partner_type].new_session(1-my_index, scenario.get_kb(1-my_index))

            controller = Controller(scenario, [my_session, partner_session], debug=False)
            self.controller_queue.put(controller)

            return controller, my_session, partner_session

        def _pair_with_human(cursor, userid, my_index, partner_id, scenario):
            controller, my_session, partner_session = _init_controller(my_index, Partner.Human, scenario)
            self.controller_map[userid] = controller
            self.controller_map[partner_id] = controller

            self.sessions[userid] = my_session
            self.sessions[partner_id] = partner_session

            next_room_id = _get_max_room_id(cursor)
            self._update_user(cursor, partner_id,
                              status=Status.Chat,
                              room_id=next_room_id,
                              partner_id=userid,
                              scenario_id=scenario.uuid,
                              agent_index=1 - my_index,
                              selected_index=-1,
                              message="")

            self._update_user(cursor, userid,
                              status=Status.Chat,
                              room_id=next_room_id,
                              partner_id=partner_id,
                              scenario_id=scenario_id,
                              agent_index=my_index,
                              selected_index=-1,
                              message="")

            return next_room_id

        def _pair_with_bot(cursor, userid, my_index, bot_type, scenario):
            controller, my_session, bot_session = _init_controller(my_index, bot_type, scenario)
            self.controller_queue.put(controller)
            self.controller_map[userid] = controller

            self.sessions[userid] = my_session

            next_room_id = _get_max_room_id(cursor)
            self._update_user(cursor, userid,
                              status=Status.Chat,
                              room_id=next_room_id,
                              partner_id=Partner.Bot,
                              scenario_id=scenario_id,
                              agent_index=my_index,
                              selected_index=-1,
                              message="")

            return next_room_id

        def _get_other_waiting_users(cursor, userid):
            cursor.execute("SELECT name FROM ActiveUsers WHERE name!=? AND status=? AND connected_status=1",
                           (userid, Status.Waiting))
            userids = [r[0] for r in cursor.fetchall()]
            return userids

        def _get_max_room_id(cursor):
            cursor.execute("SELECT MAX(room_id) FROM ActiveUsers", ())
            return cursor.fetchone()[0]

        try:
            with self.conn:
                cursor = self.conn.cursor()
                u = self._get_user_info(cursor, userid, assumed_status=Status.Waiting)
                others = _get_other_waiting_users(cursor, userid)

                partner_types = self.pairing_probabilities.keys()
                partner_probs = self.pairing_probabilities.values()

                partner_type = np.random.choice(partner_types, p=partner_probs)

                my_index = np.random.choice([0, 1])
                scenario = self.scenario_db.select_random()
                scenario_id = scenario.uuid
                if partner_type == Partner.Human:
                    if len(others) == 0:
                        return None
                    partner_id = np.random.choice(others)
                    return _pair_with_human(cursor, userid, my_index, partner_id, scenario)
                else:
                    return _pair_with_bot(cursor, userid, my_index, partner_type, scenario)

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
            self.dump_transcript(cursor, controller, userid)
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

    def create_user_if_necessary(self, username):
        with self.conn:
            cursor = self.conn.cursor()
            now = current_timestamp_in_seconds()
            logger.debug("Created user %s" % username[:6])
            cursor.execute('''INSERT OR IGNORE INTO ActiveUsers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                           (username, Status.Waiting, now, 0, now, "", -1, "", "", "", -1, -1, 0))

    def disconnect(self, userid):
        try:
            with self.conn:
                cursor = self.conn.cursor()
                self._update_user(cursor, userid,
                                  connected_status=0)

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def dump_transcript(self, cursor, controller, userid):
        user_info = self._get_user_info_unchecked(cursor, userid)
        path = os.path.join(self.config["logging"]["chat_dir"], 'Chat_%d' % user_info.room_id)
        controller.dump(path)

    def get_chat_info(self, userid):
        try:
            with self.conn:
                logger.info("Getting chat info for user %s" % userid[:6])
                cursor = self.conn.cursor()
                u = self._get_user_info_unchecked(cursor, userid)
                num_seconds_remaining = (self.config["status_params"]["chat"]["num_seconds"] +
                                         u.status_timestamp) - current_timestamp_in_seconds()
                scenario = self.scenario_db.get(u.scenario_id)
                return UserChatSession(u.room_id, u.agent_index, scenario.uuid, scenario.get_kb(u.agent_index), num_seconds_remaining)

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def get_finished_info(self, userid, from_mturk=False, completed=True):
        def _generate_mturk_code(completed=True):
            if completed:
                return "MTURK_TASK_C{}".format(str(uuid.uuid4().hex))
            return "MTURK_TASK_I{}".format(str(uuid.uuid4().hex))

        def _add_finished_task_row(cursor, userid, mturk_code, num_chats_completed):
            logger.info(
                "Adding row into CompletedTasks: userid={},mturk_code={},numsingle={},numchats={},grant_bonus={}".format(
                    userid[:6],
                    mturk_code,
                    num_chats_completed))
            cursor.execute('INSERT INTO CompletedTasks VALUES (?,?,?)',
                           (userid, mturk_code, num_chats_completed))

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
                    message = u.message

                    _add_finished_task_row(cursor, userid, mturk_code, u.num_chats_completed)
                    return FinishedSession(Markup(message), num_seconds, mturk_code)

                else:
                    return FinishedSession(Markup(u.message), num_seconds)

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def get_waiting_info(self, userid):
        try:
            with self.conn:
                logger.info("Getting waiting session info for user %s" % userid[:6])
                cursor = self.conn.cursor()
                u = self._get_user_info(cursor, userid, assumed_status=Status.Waiting)
                num_seconds = (self.config["status_params"]["waiting"][
                                   "num_seconds"] + u.status_timestamp) - current_timestamp_in_seconds()
                return WaitingSession(u.message, num_seconds)

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
                        userid[:6], Status._names[u.status]))
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

                    if u.room_id != u2.room_id or self.controller_map[userid] != self.controller_map[u.partner_id]:
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
                        Status._names[assumed_status], userid[:6]))
                    u = self._get_user_info(cursor, userid, assumed_status=assumed_status)
                    if u.status == Status.Waiting:
                        logger.debug("User %s is waiting. Checking if other users are available for chat..")
                        self.attempt_join_room(userid)
                        u = self._get_user_info(cursor, userid, assumed_status=assumed_status)
                    logger.debug("Returning TRUE (user status hasn't changed)")
                    return True
                except (UnexpectedStatusException, ConnectionTimeoutException, StatusTimeoutException) as e:
                    logger.warn(
                        "Caught %s while getting status for user %s. Returning FALSE (user status has changed)" % (
                            type(e).__name__, userid[:6]))
                    if isinstance(e, UnexpectedStatusException):
                        logger.warn("Found status %s, expected (assumed) status %s" % (
                            Status._names[e.found_status], Status._names[e.expected_status]))
                    return False
        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

    def receive(self, userid):
        session = self._get_session(userid)
        return session.poll_inbox()

    def select(self, userid, idx):
        try:
            with self.conn:
                cursor = self.conn.cursor()
                u = self._get_user_info_unchecked(cursor, userid)
                scenario = self.scenario_db.get(u.scenario_id)
                kb = scenario.get_kb(u.agent_index)
                item = kb.get_item(idx)
                self.send(userid, Event.SelectionEvent(u.agent_index,
                                                       item,
                                                       datetime.datetime.now().strftime(date_fmt)))
                return item
        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")
            return None

    def send(self, userid, event):
        session = self._get_session(userid)
        session.enqueue(event)
        controller = self.controller_map[userid]
        controller.step()

    def submit_survey(self, userid, data):
        def _user_finished(userid):
            # message = "<h3>Great, you've finished this task!</h3>"
            logger.info("Updating user %s to status FINISHED from status survey" % userid)
            self._update_user(cursor, userid, status=Status.Finished)

        try:
            with self.conn:
                cursor = self.conn.cursor()
                user_info = self._get_user_info_unchecked(cursor, userid)

                cursor.execute('INSERT INTO Surveys VALUES (?,?,?,?)',
                               (userid, user_info.partner_type, data['question1'], data['question2']))
                _user_finished(userid)
        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")
