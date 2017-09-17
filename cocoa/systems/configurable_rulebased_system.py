import random
import os
from threading import Lock
from collections import defaultdict

from cocoa.core.util import write_json, read_json, write_pickle, read_pickle
from rulebased_system import RulebasedSystem


class ConfigurableRulebasedSystem(RulebasedSystem):
    def __init__(self, timed_session, configs, policy='random', max_chats_per_config=10, db='trials.pkl'):
        """A parameterized rulebased bot that can optimize its parameters.

        Args:
            timed_session (bool)
            configs (list(tuple)): each config is a tuple of numbers/strings
            policy (str): exploration strategy
            max_chats_per_config (int)
            db (str): path to dump evaluation results

        """
        super(ConfigurableRulebasedSystem, self).__init__(timed_session)
        self.configs = configs
        self.policy = policy
        self.max_chats_per_config = max_chats_per_config
        # Save evaluation results of each config
        self.db = db
        if os.path.exists(self.db):
            self.trials = read_pickle(self.db)
        else:
            self.trials = defaultdict(dict)
        # The web server can update trials with multiple threads
        self.lock = Lock()

    @classmethod
    def name(cls):
        return 'config-rulebased'

    def update_trials(self, results):
        """Update evaluation results of configs.

        Args:
            results (list[(config, chat_id, result)])
                chat_id (str)
                result (dict): {obj (str): value (float)}
        """
        with self.lock:
            for config, chat_id, result in results:
                if not chat_id in self.trials[config]:
                    self.trials[config][chat_id] = {}
                for obj, val in result.iteritems():
                    self.trials[config][chat_id][obj] = val
            write_pickle(self.trials, self.db)

    def random_sample_config(self):
        config_counts = []
        for config in self.configs:
            count = len(self.trials[config])
            if count >= self.max_chats_per_config:
                continue
            config_counts.append((config, count))
        if len(config_counts) > 0:
            return random.choice(config_counts)[0]
        else:
            return random.choice(self.configs)

    def choose_config(self):
        """Select a config given current trials.
        """
        with self.lock:
            if self.policy == 'random':
                return self.random_sample_config()

    def new_session(self, agent, kb):
        config = self.choose_config()
        return super(ConfigurableRulebasedSystem, self).new_session(agent, kb, config=config)
