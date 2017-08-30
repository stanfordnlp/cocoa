import json
from cocoa.web.main.db_reader import DatabaseReader as BaseDatabaseReader

class DatabaseReader(BaseDatabaseReader):
    def process_event_data(self, action, data):
        if action == 'select':
            data = json.loads(data)
        return data
