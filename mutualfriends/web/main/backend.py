import sqlite3
import json
import time

from cocoa.web.main.backend import Backend as BaseBackend
from cocoa.web.main.backend import DatabaseManager as BaseDatabaseManager
from cocoa.web.main.utils import Status, Messages
from cocoa.web.views.utils import format_message
from cocoa.analysis.utils import reject_transcript

from db_reader import DatabaseReader
from core.event import Event

class DatabaseManager(BaseDatabaseManager):
    @classmethod
    def add_survey_table(cls, cursor):
        cursor.execute(
            '''CREATE TABLE survey (name text, chat_id text, partner_type text,
            cooperative integer, humanlike integer, comments text)''')

    @classmethod
    def init_database(cls, db_file):
        super(DatabaseManager, cls).init_database(db_file)
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        c.execute(
            '''CREATE TABLE bot (chat_id text, type text, config text)'''
        )
        cls.add_survey_table(c)
        conn.commit()
        conn.close()
        return cls(db_file)

    def add_scenarios(self, scenario_db, systems, update=False):
        """Add used scenarios to DB so that we don't collect data on duplicated scenarios.
        """
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        for scenario in scenario_db.scenarios_list:
            sid = scenario.uuid
            for agent_type in systems.keys():
                if update:
                    c.execute('''INSERT OR IGNORE INTO scenario VALUES (?,?, "[]", "[]")''', (sid, agent_type))
                else:
                    c.execute('''INSERT INTO scenario VALUES (?,?, "[]", "[]")''', (sid, agent_type))

        conn.commit()
        conn.close()

class Backend(BaseBackend):
    def display_received_event(self, event):
        if event.action == 'select':
            ordered_item = self.schema.get_ordered_item(event.data)
            message = format_message("Your partner selected: {}".format(", ".join([v[1] for v in ordered_item])), True)
            return {'message': message, 'status': False}
        else:
            return super(Backend, self).display_received_event(event)

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

    def should_reject_chat(self, userid, agent_idx):
        with self.conn:
            controller = self.controller_map[userid]
            cursor = self.conn.cursor()
            chat_id = controller.get_chat_id()
            ex = DatabaseReader.get_chat_example(cursor, chat_id, self.scenario_db)
            if sum([1 if event.agent == agent_idx and event.action == 'select' else 0 for event in ex.events]) > 3:
                return True
            return False

    def check_game_over_and_transition(self, cursor, userid, partner_id):
        agent_idx = self.get_agent_idx(userid)
        game_over, game_complete = self.is_game_over(userid)
        controller = self.controller_map[userid]
        chat_id = controller.get_chat_id()

        def verify_chat(userid, agent_idx, is_partner):
            user_name = 'partner' if is_partner else 'user'
            if self.should_reject_chat(userid, agent_idx):
                self.logger.debug("Rejecting chat with ID {:s} for {:s} {:s} (agent ID {:d}), and "
                                  "redirecting".format(chat_id, user_name, userid, agent_idx))
                self.end_chat_and_redirect(cursor, userid,
                                           message=self.messages.Redirect + " " + self.messages.Waiting)
            else:
                msg, _ = self.get_completion_messages(userid)
                self.logger.debug("Accepted chat with ID {:s} for {:s} {:s} (agent ID {:d}), and redirecting to "
                                  "survey".format(chat_id, user_name, userid, agent_idx))
                self.end_chat_and_finish(cursor, userid, message=msg)

        if game_over:
            if not self.is_user_partner_bot(cursor, userid):
                verify_chat(partner_id, 1 - agent_idx, True)
            verify_chat(userid, agent_idx, False)
            return True

        return False

    def submit_survey(self, userid, data):
        def _user_finished(userid):
            self._update_user(cursor, userid, status=Status.Finished)

        def _update_scenario_db(chat_id, scenario_id, partner_type):
            # make sure that the # of completed dialogues for the scenario is only updated once if both agents are human
            cursor.execute('''SELECT complete FROM scenario WHERE scenario_id=? AND partner_type=?''',
                           (scenario_id, partner_type))
            complete_set = set(json.loads(cursor.fetchone()[0]))
            complete_set.add(chat_id)
            cursor.execute('''
                UPDATE scenario
                SET complete=?
                WHERE scenario_id=? AND partner_type=?
                AND (SELECT COUNT(survey.name)
                    FROM survey
                    WHERE survey.chat_id=?) = 0;
            ''', (json.dumps(list(complete_set)), scenario_id, partner_type, chat_id))

        try:
            with self.conn:
                cursor = self.conn.cursor()
                user_info = self._get_user_info_unchecked(cursor, userid)
                cursor.execute('''SELECT scenario_id FROM chat WHERE chat_id=?''', (user_info.chat_id,))
                scenario_id = cursor.fetchone()[0]
                _update_scenario_db(user_info.chat_id, scenario_id, user_info.partner_type)
                cursor.execute('INSERT INTO survey VALUES (?,?,?,?,?,?)',
                               (userid, user_info.chat_id, user_info.partner_type,
                                data['cooperative'], data['humanlike'], data['comments']))
                _user_finished(userid)
                self.logger.debug("User {:s} submitted survey for chat {:s}".format(userid, user_info.chat_id))

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")

#######################################
from flask import g
from flask import current_app as app
from utils import Messages

def get_backend():
    backend = getattr(g, '_backend', None)
    if backend is None:
        g._backend = Backend(app.config["user_params"],
                         app.config["schema"],
                         app.config["scenario_db"],
                         app.config["systems"],
                         app.config["sessions"],
                         app.config["controller_map"],
                         app.config["pairing_probabilities"],
                         app.config["num_chats_per_scenario"],
                         Messages)
        backend = g._backend
    return backend
