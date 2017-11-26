import sqlite3
from datetime import datetime
import json

from cocoa.core.dataset import Example
from cocoa.core.event import Event
from cocoa.core.util import write_json

class DatabaseReader(object):
    date_fmt = '%Y-%m-%d %H-%M-%S'

    @classmethod
    def convert_time_format(cls, time):
        if time is None:
            return time
        try:
            dt = datetime.strptime(time, cls.date_fmt)
            s = str((dt - datetime.fromtimestamp(0)).total_seconds())
            return s
        except (ValueError, TypeError):
            try:
                dt = datetime.fromtimestamp(float(time)) # make sure that time is a UNIX timestamp
                return time
            except (ValueError, TypeError):
                print 'Unrecognized time format: %s' % time

        return None

    @classmethod
    def process_event_data(cls, action, data):
        """Construct structured data from strings.

        Data can be some json dict string that needs to be loaded,
        e.g. "{'price': 10}".

        """
        if action == 'eval':
            data = json.loads(data)
        return data

    @classmethod
    def get_chat_outcome(cls, cursor, chat_id):
        """Get outcome of the chat specified by chat_id.

        Returns:
            {}

        """
        cursor.execute('SELECT outcome FROM chat WHERE chat_id=?', (chat_id,))
        outcome = cursor.fetchone()[0]
        try:
            outcome = json.loads(outcome)
        except ValueError:
            outcome = {'reward': 0}
        return outcome

    @classmethod
    def get_chat_agent_types(cls, cursor, chat_id):
        """Get types of the two agents in the chat specified by chat_id.

        Returns:
            {0: agent_name (str), 1: agent_name (str)}

        """
        try:
            cursor.execute('SELECT agent_types FROM chat WHERE chat_id=?', (chat_id,))
            agent_types = cursor.fetchone()[0]
            agent_types = json.loads(agent_types)
        except sqlite3.OperationalError:
            agent_types = {0: HumanSystem.name(), 1: HumanSystem.name()}
        return agent_types

    @classmethod
    def get_chat_events(cls, cursor, chat_id):
        """Read all events in the chat specified by chat_id.

        Returns:
            [Event]

        """
        cursor.execute('SELECT * FROM event WHERE chat_id=? ORDER BY time ASC', (chat_id,))
        logged_events = cursor.fetchall()

        chat_events = []
        agent_chat = {0: False, 1: False}
        for row in logged_events:
            # Compatible with older event structure
            agent, action, time, data = [row[k] for k in ('agent', 'action', 'time', 'data')]
            try:
                start_time = row['start_time']
            except IndexError:
                start_time = time
            try:
                metadata = json.loads(row['metadata'])
            except IndexError:
                metadata = None

            if action == 'join' or action == 'leave' or action == 'typing':
                continue
            if action == 'message' and len(data.strip()) == 0:
                continue

            data = cls.process_event_data(action, data)
            agent_chat[agent] = True
            time = cls.convert_time_format(time)
            start_time = cls.convert_time_format(start_time)
            event = Event(agent, time, action, data, start_time, metadata=metadata)
            chat_events.append(event)

        return chat_events

    @classmethod
    def has_chat(cls, cursor, chat_id):
        """Check if a chat is in the DB.
        """
        cursor.execute('SELECT scenario_id, outcome FROM chat WHERE chat_id=?', (chat_id,))
        result = cursor.fetchone()
        if result is None:
            return False
        return True

    @classmethod
    def get_chat_scenario_id(cls, cursor, chat_id):
        cursor.execute('SELECT scenario_id FROM chat WHERE chat_id=?', (chat_id,))
        uuid = cursor.fetchone()[0]
        return uuid

    @classmethod
    def get_chat_example(cls, cursor, chat_id, scenario_db):
        """Read a dialogue from the DB.

        Args:
            chat_id (str)
            scenario_db (ScenarioDB): map scenario ids to Scenario

        Returns:
            Example

        """
        if not cls.has_chat(cursor, chat_id):
            return None

        scenario_uuid = cls.get_chat_scenario_id(cursor, chat_id)
        scenario = scenario_db.get(scenario_uuid)
        events = cls.get_chat_events(cursor, chat_id)
        outcome = cls.get_chat_outcome(cursor, chat_id)
        agent_types = cls.get_chat_agent_types(cursor, chat_id)

        return Example(scenario, scenario_uuid, events, outcome, chat_id, agent_types)

    @classmethod
    def dump_chats(cls, cursor, scenario_db, json_path, uids=None):
        """Dump chat transcripts to a JSON file.

        Args:
            scenario_db (ScenarioDB): retrieve Scenario by logged uuid.
            json_path (str): output path.
            uids (list): if provided, only log chats from these users.

        """
        if uids is None:
            cursor.execute('SELECT DISTINCT chat_id FROM event')
            ids = cursor.fetchall()
        else:
            ids = []
            uids = [(x,) for x in uids]
            for uid in uids:
                cursor.execute('SELECT chat_id FROM mturk_task WHERE name=?', uid)
                ids_ = cursor.fetchall()
                ids.extend(ids_)

        def is_single_agent(chat):
            agent_event = {0: 0, 1: 0}
            for event in chat.events:
                agent_event[event.agent] += 1
            return agent_event[0] == 0 or agent_event[1] == 0

        examples = []
        for chat_id in ids:
            ex = cls.get_chat_example(cursor, chat_id[0], scenario_db)
            if ex is None or is_single_agent(ex):
                continue
            examples.append(ex)

        write_json([ex.to_dict() for ex in examples], json_path)
