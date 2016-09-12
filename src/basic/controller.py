from util import generate_uuid
from dataset import Example

class Controller(object):
    '''
    The controller takes two systems and can run them to generate a dialgoue.
    '''
    def __init__(self, scenario, systems):
        self.scenario = scenario
        self.systems = systems
        assert len(self.systems) == 2
        for agent in (0, 1):
            self.scenario.kbs[agent].dump()

    def run(self):
        '''Simulate the dialogue.'''
        events = []
        time = 0
        selections = [None, None]
        reward = 0
        game_over = False
        for it in range(100):
            for agent, system in enumerate(self.systems):
                event = system.send()
                time += 1
                if not event:
                    continue
                event.time = time
                events.append(event)

                if event.action == 'select':
                    selections[agent] = event.data

                print 'agent=%s: system=%s, event=%s' % (agent, type(system).__name__, event.to_dict())
                for partner, other_system in enumerate(self.systems):
                    if agent != partner:
                        other_system.receive(event)

                # Game is over when the two selections are the same
                if selections[0] is not None and selections[0] == selections[1]:
                    reward = 1
                    game_over = True
                    break
            if game_over:
                break

        uuid = generate_uuid('E')
        outcome = {'reward': reward}
        print 'outcome: %s' % outcome
        return Example(self.scenario, uuid, events, outcome)
