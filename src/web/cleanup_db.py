__author__ = 'anushabala'

import sqlite3
import time
import atexit
import json
from argparse import ArgumentParser
from src.basic.systems.human_system import HumanSystem
import multiprocessing


class DBCleaner():
    SLEEP_TIME = 15

    def __init__(self, db_file, chat_timeout, user_timeout):
        self.db_file = db_file
        self.chat_timeout = chat_timeout
        self.user_timeout = user_timeout
        self.cleaned_chats = set()
        self._stop = False

    @staticmethod
    def cleanup(db_file, chat_timeout, user_timeout, cleaned_chats, sleep_time, q):
        def _cleanup_corrupt_counts():
            # this should never happen!
            cursor.execute('''UPDATE scenario SET active=0 WHERE active < 0''')

        def _update_inactive_chats(chats):
            for chat_info in chats:
                chat_id, sid, outcome, _, agent_types, start_time = chat_info

                if chat_id not in cleaned_chats:
                    # if it's been longer than chat_timeout seconds since the chat started, and the chat
                    # wasn't previously cleaned up, update the scenario DB

                    agent_types = json.loads(agent_types)
                    partner_type = agent_types['0'] if agent_types['1'] == HumanSystem.name() else agent_types['1']
                    print "[Cleaner] Cleaned up chat with ID={}, partner_type={}, scenario_id={}".format(
                        chat_id, partner_type, sid
                    )
                    cursor.execute('''
                    UPDATE scenario SET active=active-1 WHERE partner_type=? AND scenario_id=?
                    ''', (partner_type, sid))

                    cleaned_chats.add(chat_id)

        def _find_incomplete_chats():
            # print 'Finding timed out chats with no outcome'
            cursor.execute('''SELECT * FROM chat WHERE outcome="" AND start_time <?''', (now-chat_timeout,))
            # Select all incomplete chats (with empty outcomes) that have timed out
            return cursor.fetchall()

        def _is_connection_timed_out(userid):
            cursor.execute('''SELECT connected_status, connected_timestamp FROM active_user WHERE name=?''', (userid,))
            status, tmstp = cursor.fetchone()
            if status == 0 and tmstp < now - user_timeout:
                return True

            return False

        def _find_disconnected_user_chats():
            """
            Find chats with no outcome where at least one human agent has been disconnected longer
            than user_timeout seconds
            :return:
            """
            # print 'Finding chats with no outcome and users with timed out connections'
            cursor.execute('''SELECT * FROM chat WHERE outcome=""''')
            inc_chats = cursor.fetchall()

            disconnected_chats = []
            for chat_info in inc_chats:
                chat_id, sid, outcome, agent_ids, agent_types, start_time = chat_info
                agent_types = json.loads(agent_types)
                agent_ids = json.loads(agent_ids)
                human_idxes = [k for k in agent_types.keys() if agent_types[k] == HumanSystem.name()]
                clean = False
                for idx in human_idxes:
                    userid = agent_ids[idx]
                    if _is_connection_timed_out(userid):
                        # print "User %s connection timeout" % userid
                        clean = True

                if clean:
                    disconnected_chats.append(chat_info)
            return disconnected_chats

        try:
            conn = sqlite3.connect(db_file)
            with conn:
                cursor = conn.cursor()
                now = time.time()

                filters = [_find_incomplete_chats, _find_disconnected_user_chats]
                for f in filters:
                    chats = f()
                    _update_inactive_chats(chats)

                _cleanup_corrupt_counts()
                q.put(cleaned_chats)

                # print "[Cleaner] Sleeping for %d seconds" % sleep_time
            time.sleep(sleep_time)

        except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")
        except KeyboardInterrupt:
            if q.empty():
                q.put(cleaned_chats)

    def cancel(self):
        print "[Cleaner] Stopping execution"
        self._stop = True

    def stopped(self):
        return self._stop

    def start(self):
        print "[Cleaner] Starting execution"
        while not self.stopped():
            q = multiprocessing.Queue()
            p = multiprocessing.Process(target=self.cleanup, args=(self.db_file, self.chat_timeout, self.user_timeout, self.cleaned_chats, self.SLEEP_TIME, q))
            try:
                p.start()
                p.join()
                self.cleaned_chats = q.get()
                # print "[Cleaner] Awoke from sleep"
                # print "[Cleaner] Cleaned chats from queue:", self.cleaned_chats

            except KeyboardInterrupt:
                # If program is killed, try to run cleanup one last time in case past run was interrupted
                p.join()
                self.cancel()
                if not q.empty():
                    # print "[Cleaner] Got item from queue from killed process"
                    self.cleaned_chats = q.get()
                    # print self.cleaned_chats
                p = multiprocessing.Process(target=self.cleanup, args=(self.db_file, self.chat_timeout, self.user_timeout, self.cleaned_chats, 0, q))
                p.start()
                p.join()

        print "[Cleaner] Stopped execution"


def stop_cleanup(handler):
    handler.cancel()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', help='Path to config file for website')
    parser.add_argument('--db-file', help='Path to DB file to cleanup')

    args = parser.parse_args()
    params = json.load(open(args.config, 'r'))
    cleanup_handler = DBCleaner(args.db_file,
                                chat_timeout=params['status_params']['chat']['num_seconds'] + 30,
                                user_timeout=params['connection_timeout_num_seconds'] + 5)
    # atexit.register(stop_cleanup, handler=cleanup_handler)
    cleanup_handler.start()




