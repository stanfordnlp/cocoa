import sqlite3
import json
import time

from cocoa.web.main.backend import Backend as BaseBackend
from cocoa.web.main.backend import DatabaseManager as BaseDatabaseManager
from cocoa.web.main.utils import Status, Messages

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
    def select(self, userid, idx):
        try:
            with self.conn:
                cursor = self.conn.cursor()
                u = self._get_user_info_unchecked(cursor, userid)
                scenario = self.scenario_db.get(u.scenario_id)
                kb = scenario.get_kb(u.agent_index)
                item = kb.items[idx]
                print 'selected:', item
                self.send(userid, Event.SelectionEvent(u.agent_index,
                                                       item,
                                                       str(time.time())))
                return item
        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")
            return None

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
