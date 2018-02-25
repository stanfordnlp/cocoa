class Utterance(object):
    def __init__(self, raw_text=None, tokens=None, logical_form=None, template=None, ambiguous_template=False, agent=None):
        self.text = raw_text
        self.tokens = tokens
        self.lf = logical_form
        self.template = template
        self.ambiguous_template = ambiguous_template
        self.agent = agent

    def to_dict(self):
        return {
                'logical_form': self.lf.to_dict(),
                'template': self.template,
                }

    def __str__(self):
        s = []
        s.append('-'*20)
        if self.text:
            s.append(self.text)
        if self.lf:
            s.append(self.lf)
        if self.template:
            s.append(' '.join(self.template))
        return '\n'.join([str(x) for x in s])

class LogicalForm(object):
    def __init__(self, intent, **kwargs):
        self.intent = intent
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    def to_dict(self):
        attrs = vars(self)
        attrs['intent'] = self.intent
        return attrs

    def __str__(self):
        attrs = vars(self)
        s = ' '.join(['{}={}'.format(k, v) for k, v in attrs.iteritems()])
        return s

class Parser(object):
    greeting_words = set(['hi', 'hello', 'hey', 'hiya', 'howdy'])

    question_words = set(['what', 'when', 'where', 'why', 'which', 'who', 'whose', 'how', 'do', 'did', 'does', 'are', 'is', 'would', 'will', 'can', 'could', 'any'])

    neg_words = set(['no', 'not', "n't"])

    def __init__(self, agent, kb, lexicon):
        self.agent = agent
        self.partner = 1 - agent
        self.kb = kb
        self.lexicon = lexicon

    @classmethod
    def is_negative(cls, utterance):
        for token in utterance.tokens:
            if token in cls.neg_words:
                return True
        return False

    @classmethod
    def is_question(cls, utterance):
        tokens = utterance.tokens
        if len(tokens) < 1:
            return False
        last_word = tokens[-1]
        first_word = tokens[0]
        return last_word == '?' or first_word in cls.question_words

    @classmethod
    def is_greeting(cls, utterance):
        for token in utterance.tokens:
            if token in cls.greeting_words:
                return True
        return False

    def tag_utterance(self, utterance):
        """Tag the utterance with basic speech acts.
        """
        tags = []
        if self.is_question(utterance):
            tags.append('question')
        if self.is_greeting(utterance):
            tags.append('greeting')
        if self.is_negative(utterance):
            tags.append('negative')
        return tags

    def parse(self, event, dialogue_state, update_state=False):
        """Parse an event to LogicalForm.
        """
        raise NotImplementedError

    def parse_action(self, event):
        intent = event.action
        return Utterance(logical_form=LogicalForm(intent), template=['<{}>'.format(intent)])

