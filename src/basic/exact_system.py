from simple_system import SimpleSystem
from collections import defaultdict

class ExactSystem(SimpleSystem):
    def __init__(self, agent, kb):
        self.agent = agent
        self.kb = kb
        self.response = None
        self.attributes = [attr.name for attr in self.kb.schema.attributes]
        self.matched_item = None
        self.selected = False
        # Recursion stack
        self.stack = []
        init_state = {'items': self.kb.items,\
                'attrs': self.init_attr_state(self.kb.items),\
                'fixed_attrs': None,\
                }
        self.stack.append(init_state)

    def print_current_items(self):
        print 'current items:'
        for item in self.stack[-1]['items']:
            print item.values()

    def receive(self, event):
        print 'RECEIVE'
        if event.action == 'message':
            self.response = self.receive_constraint(event.data)
        elif event.action == 'select':
            for item in self.kb.items:
                if item == event.data:
                    self.matched_item = item
        self.print_current_items()

    def select_attr(self, attr_state):
        '''
        Select the next attribute and value to take action on.
        Heuristic: select the attribute with the least number of values
        '''
        min_num_values = 99999
        attr_name = None
        for name, values in attr_state.iteritems():
            n = len(values)
            if n < min_num_values:
                attr_name = name
                min_num_values = n
        if not attr_name:
            return None, None
        print 'select attr:', attr_name
        # Select the value with the highest count
        max_count = 0
        attr_value = None
        for value, status in attr_state[attr_name].iteritems():
            #print value, status
            if status['checked'] is False and status['count'] > max_count:
                max_count = status['count']
                attr_value = value
        # All values of this attributes have been checked
        if attr_value is None:
            print 'all values checked. re-select.'
            del attr_state[attr_name]
            return self.select_attr(attr_state)
        print 'selected:', attr_name, attr_value
        return attr_name, attr_value

    def init_attr_state(self, items, fixed_attr=None):
        '''
        state[attr_name][attr_value] = (count, checked)
        '''
        state = defaultdict(lambda : defaultdict(lambda : {'count': 0, 'checked': False}))
        for item in items:
            for name, value in item.iteritems():
                if name != fixed_attr:
                    state[name][value]['count'] += 1

        # Delete attributes checked in parent state
        if len(self.stack) > 0:
            prev_state = self.stack[-1]['attrs']
            checked_attrs = []
            for name, values in state.iteritems():
                if name not in prev_state:
                    checked_attrs.append(name)
            for name in checked_attrs:
                del state[name]
        return state

    def add_constraint(self):
        curr_items = self.stack[-1]['items']
        curr_attrs = self.stack[-1]['attrs']
        name, value = self.select_attr(curr_attrs)
        print 'add constraint:', name, value
        # TODO: consider currently checking attributes
        if name is None:
            if len(curr_items) == 1:
                print 'one item left. select:', curr_items[0]
                return self.select(curr_items[0])
            else:
                raise Exception('No available attribute to select')
        else:
            new_items = [item for item in curr_items if item[name] == value]
            assert len(new_items) > 0
            print 'push to stack %d items' % len(new_items)
            curr_attrs[name][value]['checked'] = True
            new_state = {'items': new_items,\
                    'attrs': self.init_attr_state(new_items, name),\
                    'fixed_attrs': (name, value),\
                    }
            self.stack.append(new_state)
            # Whether all of my current items satisfy this constraint
            # This information is useful for early stopping
            all_ = True if len(new_items) == len(curr_items) else False
            return self.inform((name, value, True), len(new_items), all_)

    def receive_constraint(self, constraint):
        (name, value, exist), num, all_ = constraint
        print 'receive constraint:', constraint
        curr_items = self.stack[-1]['items']
        curr_attrs = self.stack[-1]['attrs']
        if exist:
            new_items = [item for item in curr_items if item[name] == value]
            n = len(new_items)
            if n > 0:
                # TODO: partner has one item with attr, all of my items have attr, should inform partner to select that one
                if all_:
                    # Other values are not possible
                    print 'all. remove items %s != %s' % (name, value)
                    self.stack[-1]['items'] = new_items
                print 'push to stack %d items' % n
                # Checked by partner
                curr_attrs[name][value]['checked'] = True
                new_state = {'items': new_items,\
                        'attrs': self.init_attr_state(new_items, name),\
                        'fixed_attrs': (name, value),\
                        }
                self.stack.append(new_state)
            elif n == 0:
                if all_:
                    print 'invalid constraint. received True.'
                    self.pop_stack()
                else:
                    print 'invalid constraint. inform.'
                return self.inform((name, value, False))
        else:
            print 'invalid constraint. received False.'
            self.pop_stack()
            return None

    def pop_stack(self):
        while True:
            print 'pop stack'
            state = self.stack.pop()
            fixed_attr = state['fixed_attrs']
            if fixed_attr is not None:
                name, value = fixed_attr
                # Remove unsatisfied item in parent
                new_items = [item for item in self.stack[-1]['items'] if item[name] != value]
                self.stack[-1]['items'] = new_items
                if len(new_items) > 0:
                    break
            else:
                # We are at the root
                break

    def inform(self, cond, n=0, all_=False):
        print 'inform:', cond, n, all_
        return self.message((cond, n, all_))

    def send(self):
        print 'SEND'

        # Check global possible items: if the set has been reduced to one item, select that.
        if len(self.stack[0]['items']) == 1:
            self.matched_item = self.stack[-1]['items'][0]

        # We found a match (note that this doesn't always work)
        if self.matched_item and not self.selected:
            self.selected = True
            print 'select:', self.matched_item
            return self.select(self.matched_item)

        # Inform
        if self.response is not None:
            # response given by receive
            message = self.response
            self.response = None
        else:
            message = self.add_constraint()

        self.print_current_items()
        return message


