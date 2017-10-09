import sqlite3
import json
import time

from cocoa.web.main.backend import Backend as BaseBackend
from cocoa.web.main.backend import DatabaseManager as BaseDatabaseManager
from cocoa.web.main.utils import Status, Messages
from cocoa.analysis.utils import reject_transcript

from db_reader import DatabaseReader
from core.event import Event
# from analysis.analyze_strategy import StrategyAnalyzer

class DatabaseManager(BaseDatabaseManager):
    @classmethod
    def add_survey_table(cls, cursor):
        cursor.execute(
            '''CREATE TABLE survey (chat_id text, negotiator integer, comments text)''')

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
    def should_reject_chat(self, userid, agent_idx):
        with self.conn:
            controller = self.controller_map[userid]
            cursor = self.conn.cursor()
            chat_id = controller.get_chat_id()
            ex = DatabaseReader.get_chat_example(cursor, chat_id, self.scenario_db).to_dict()
            return reject_transcript(ex, agent_idx, min_tokens=40)

    def get_margin(self, controller, agent_idx):
        with self.conn:
            cursor = self.conn.cursor()
            chat_id = controller.get_chat_id()
            ex = DatabaseReader.get_chat_example(cursor, chat_id, self.scenario_db)
            outcome = controller.get_outcome()
            role = ex.scenario.kbs[agent_idx].facts['personal']['Role']
            if outcome['reward'] == 0:
                return role, None
            else:
                try:
                    price = float(outcome['offer']['price'])
                except (KeyError, ValueError) as e:
                    return role, None
                # margin = StrategyAnalyzer.get_margin(ex, price, agent_idx, role, remove_outlier=False)
                margin = 7  # this is a stub to allow the server to run
                return role, margin

    def check_game_over_and_transition(self, cursor, userid, partner_id):
        agent_idx = self.get_agent_idx(userid)
        game_over, game_complete = self.is_game_over(userid)
        controller = self.controller_map[userid]
        chat_id = controller.get_chat_id()

        if game_over:
            # TODO: message
            msg, _ = self.get_completion_messages(userid)
            self.end_chat_and_finish(cursor, userid, message=msg)
            return True

        return False

    def get_completion_messages(self, userid):
        """
        Returns two completion messages: one for the current user and one for the user's partner. This function doesn't
        check whether the user's partner is a bot or not. It just decides how many points the user gets and reports
        that score accordingly, along with the completion message
        """

        _, game_complete = self.is_game_over(userid)
        if game_complete:
            msg = self.messages.ChatCompleted
            partner_msg = msg
        else:
            msg = self.messages.ChatIncomplete
            partner_msg = msg

        return msg, partner_msg

    def select(self, userid, proposal):
        try:
            with self.conn:
                cursor = self.conn.cursor()
                u = self._get_user_info_unchecked(cursor, userid)
                self._update_user(cursor, userid, connected_status=1)
                self.send(userid, Event.SelectEvent(u.agent_index,
                                                    proposal,
                                                    str(time.time()) ))
        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")
            return None

    # def reject_offer(self, userid):
    #     try:
    #         with self.conn:
    #             cursor = self.conn.cursor()
    #             u = self._get_user_info_unchecked(cursor, userid)
    #             self._update_user(cursor, userid, connected_status=1)
    #             self.send(userid, Event.RejectEvent(u.agent_index,
    #                                                str(time.time())))
    #     except sqlite3.IntegrityError:
    #         print("WARNING: Rolled back transaction")
    #         return None

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
                cursor.execute('INSERT INTO survey VALUES (?,?,?)',
                                (user_info.chat_id, data['negotiator'], data['comments']))
                _user_finished(userid)
                self.logger.debug("User {:s} submitted survey for chat {:s}".format(userid, user_info.chat_id))

                if user_info.partner_type == 'config-rulebased':
                    agent_idx = self.get_agent_idx(userid)
                    bot_agent_idx = 1 - agent_idx
                    controller = self.controller_map[userid]
                    # TODO: get config
                    cursor.execute('''SELECT config FROM bot WHERE chat_id=? AND type=?''', (user_info.chat_id, user_info.partner_type))
                    config = tuple(json.loads(cursor.fetchone()[0]))
                    role, margin = self.get_margin(controller, bot_agent_idx)
                    self.logger.debug("Updating trials for user {}".format(userid))
                    self.logger.debug("scenario_id={}, role={}, margin={}, humanlike={}".format(scenario_id, role, margin, data['negotiator']))
                    self.systems['config-rulebased'].update_trials([
                        (config, user_info.chat_id, {'scenario_id': scenario_id, 'role': role, 'margin': margin, 'humanlike': data['negotiator']}),
                        ])
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
