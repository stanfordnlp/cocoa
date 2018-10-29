from cocoa.systems.configurable_rulebased_system import ConfigurableRulebasedSystem as BaseConfigurableRulebasedSystem
from sessions.rulebased_session import RulebasedSession, Config

def add_configurable_rulebased_arguments(parser):
    parser.add_argument('--rulebased-configs')
    parser.add_argument('--config-search-policy', default='random')
    parser.add_argument('--chats-per-config', default=10, type=int)
    parser.add_argument('--trials-db', default='trials.json')

class ConfigurableRulebasedSystem(BaseConfigurableRulebasedSystem):
    def __init__(self, configs, lexicon, timed_session=False, policy='random', max_chats_per_config=10, db='trials.json', templates=None):
        configs = [tuple(c) for c in configs]
        super(ConfigurableRulebasedSystem, self).__init__(timed_session, configs, policy=policy, max_chats_per_config=max_chats_per_config, db=db)
        self.lexicon = lexicon
        self.templates = templates

    def _new_session(self, agent, kb, config):
        config = Config(*config)
        return RulebasedSession.get_session(agent, kb, self.lexicon, config, self.templates)


############# TEST ##############

if __name__ == '__main__':
    from cocoa.core.util import read_json
    configs = read_json('data/rulebased_configs.json')
    configs = [tuple(c) for c in configs]
    s = BaseConfigurableRulebasedSystem(False, configs)
    #print s.configs
    #print s.choose_config()
    #print s.choose_config()
    print s.trials
    #s.update_trials([(configs[0], '0', {'margin': 0.1, 'humanlike': 1})])
    #s.update_trials([(configs[0], '1', {'margin': 0.1, 'humanlike': 1})])
    s.update_trials([(configs[1], '3', {'margin': 0.1})])
    #s.update_trials([(configs[1], '2', {'humanlike': 0.1})])
    print s.trials
