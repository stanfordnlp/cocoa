from simple_system import SimpleSystem
from collections import defaultdict
from copy import deepcopy

class ExactSystem(SimpleSystem):
    def __init__(self, agent, kb):
        self.agent = agent
        self.kb = kb
        self.response = None
        self.attributes = [attr.name for attr in self.kb.schema.attributes]
        self.matched_item = None
        self.selected = False
        # Recursion stack
        self.item_stack = [self.kb.items]
        self.attr_stack = []
        self.attr_stack.append(self.init_attr_state(self.kb.items))

    def receive(self, event):
        print 'RECEIVE'
        if event.action == 'message':
            self.response = self.receive_constraint(event.data)
        elif event.action == 'select':
            for item in self.kb.items:
                if item == event.data:
                    self.matched_item = item

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

    def init_attr_state(self, items):
        '''
        state[attr_name][attr_value] = (count, checked)
        '''
        state = defaultdict(lambda : defaultdict(lambda : {'count': 0, 'checked': False}))
        for item in items:
            for name, value in item.iteritems():
                state[name][value]['count'] += 1
        if len(self.attr_stack) > 0:
            prev_state = self.attr_stack[-1]
            checked_attrs = []
            for name, values in state.iteritems():
                # attr checked in parent state
                if name not in prev_state:
                    checked_attrs.append(name)
                    continue
                for value in values:
                    if prev_state[name][value]['checked'] is True:
                        state[name][value]['checked'] = True
            for name in checked_attrs:
                del state[name]
        #print 'init attr state:'
        #for name in state:
        #    print 'attr:', name
        #    for value in state[name]:
        #        print 'value:', value, state[name][value]
        return state

    def add_constraint(self):
        curr_items = self.item_stack[-1]
        curr_attrs = self.attr_stack[-1]
        name, value = self.select_attr(self.attr_stack[-1])
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
            self.item_stack.append(new_items)
            self.attr_stack.append(self.init_attr_state(new_items))
            # Whether all of my current items satisfy this constraint
            # This information is useful for early stopping
            all_ = True if len(new_items) == len(curr_items) else False
            return self.inform((name, value, True), len(new_items), all_)

    def receive_constraint(self, constraint):
        (name, value, exist), num, all_ = constraint
        print 'receive constraint:', constraint
        curr_items = self.item_stack[-1]
        curr_attrs = self.attr_stack[-1]
        if exist:
            new_items = [item for item in curr_items if item[name] == value]
            n = len(new_items)
            if n > 0:
                # All vs One
                if num == len(self.kb.items) and n == 1:
                    print 'all vs one. select:', new_items[0]
                    return self.select(new_items[0])
                elif num == 1 and len(new_items) == len(self.kb.items):
                    print 'all vs one. inform.'
                    return self.inform((name, value, exist), len(new_items))
                print 'push to stack %d items' % n
                # Checked by partner
                curr_attrs[name][value]['checked'] = True
                self.item_stack.append(new_items)
                self.attr_stack.append(self.init_attr_state(new_items))
            elif n == 0:
                if all_:
                    print 'invalid constraint. pop.'
                    self.item_stack.pop()
                    self.attr_stack.pop()
                else:
                    print 'invalid constraint. inform.'
                return self.inform((name, value, False))
        else:
            print 'invalid constraint.'
            assert len(self.item_stack) > 0
            n = len(self.item_stack[-1])
            while len(self.item_stack) > 0 and \
                    len(self.item_stack[-1]) == n:
                print 'pop stack'
                self.item_stack.pop()
                self.attr_stack.pop()
            return None

    def inform(self, cond, n=0, all_=False):
        print 'inform:', cond, n, all_
        return self.message((cond, n, all_))

    def send(self):
        print 'SEND'
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
        return message


