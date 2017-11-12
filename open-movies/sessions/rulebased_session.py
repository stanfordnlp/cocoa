import random
import copy
import sys
import re
import operator
from collections import namedtuple, defaultdict

from cocoa.core.entity import is_entity, Entity

from session import Session
from core.tokenizer import tokenize

class RulebasedSession(Session):
    question_words = set(['what', 'when', 'where', 'why', 'which', 'who', 'whose', 'how', 'do', 'does', 'are', 'is', 'would', 'will', 'can', 'could'])
    greeting_words = set(['hi', 'hello', 'hey', 'hiya', 'howdy'])

    def __init__(self, agent, kb, lexicon, config, templates):
        super(RulebasedSession, self).__init__(agent)
        self.kb = kb
        self.lexicon = lexicon
        self.templates = templates
        self.known_movies = self.templates.known_movies()

        self.state = {
                'my_act': ('<start>', None),
                'partner_act': ('<start>', None),
                'curr_movie': None,
                'movie_informed': defaultdict(list),
                }

        self.used_templates = set()

    def is_question(self, tokens):
        if len(tokens) < 1:
            return False
        last_word = tokens[-1].lower()
        first_word = tokens[0].lower()
        return last_word == '?' or first_word in self.question_words

    def is_greeting(self, tokens):
        for token in tokens:
            if token.lower() in self.greeting_words:
                return True
        return False

    def is_plot(self, tokens):
        s = ' '.join(tokens)
        if re.search(r'(movie|show|story|film|drama) about', s):
            return True
        return False

    def is_opinion(self, tokens):
        s = ' '.join(tokens)
        if re.search(r'(do|did) you (like|think|enjoy|feel)', s):
            return True
        return False

    def is_bye(self, tokens):
        s = ' '.join(tokens)
        if 'bye' in tokens or re.search(r'(great|nice|fun) (talking|chatting)', s):
            return True
        return False

    # TODO: factorize parser
    # TODO: use enumerator for tags
    def receive(self, event):
        if event.action == 'done':
            self.state['partner_act'] = ('done', None)
        elif event.action == 'message':
            tokens = tokenize(event.data, lowercase=False)
            entity_tokens = self.lexicon.link_entity(tokens)
            print entity_tokens
            entities = [token for token in entity_tokens if is_entity(token)]
            if self.is_greeting(tokens):
                tag = 'greet'
            elif self.is_bye(tokens):
                tag = 'bye'
            elif self.is_question(tokens):
                if self.is_plot(tokens):
                    tag = 'ask-plot'
                elif self.is_opinion(tokens):
                    tag = 'ask-opinion'
                else:
                    tag = 'ask'
            elif len(entities) > 0:
                tag = 'inform'
            else:
                tag = 'unknown'
            self.state['partner_act'] = (tag, entities)
            print 'RECEIVE'
            print self.state['partner_act']
            known_entities = [e for e in entities if e.canonical.type == 'title']
            unknown_entities = [e for e in entities if e.canonical.type == 'unknown']
            if known_entities:
                self.state['curr_movie'] = known_entities[0].canonical.value
            elif unknown_entities:
                self.state['curr_movie'] = unknown_entities[0].canonical.value

    def choose_template(self, tag, context_tag=None, movie_title=None):
        print 'choose template: tag={tag}, context_tag={context_tag}, title={title}'.format(tag=tag, context_tag=context_tag, title=movie_title)
        template = self.templates.search(tag=tag, context_tag=context_tag, movie_title=movie_title, used_templates=self.used_templates)
        if template is None:
            return None
        self.used_templates.add(template['id'])
        template = template.to_dict()
        return template

    #def fill_template(self, template):
    #    if '{title}' in template:
    #        title = random.choice(self.known_movies)
    #        return template.format(title=title)
    #    return template

    def ask(self, context_tag=None):
        print 'ask', context_tag
        self.state['my_act'] = ('ask', None)
        response = self.choose_template(tag='ask', context_tag=context_tag)['template']
        return self.message(response)

    def start_movie(self):
        if len(self.state['movie_informed']) > 0:
            tag = 'start_another_movie'
        else:
            tag = 'start_movie'
        template = self.choose_template(tag=tag)['template']
        title = random.choice(list(self.known_movies))
        utterance = template.format(title=title.title())
        self.state['curr_movie'] = title
        self.state['my_act'] = (tag, [Entity.from_elements(surface=title, type='title')])
        return self.message(utterance)

    def inform(self, title):
        # if len(self.state['movie_informed'][title]) > 0:
        #     tag = 'inform-middle'
        # else:
        #     tag = 'inform-first'
        if partner_act == 'ask-plot':
            tag = 'inform-plot'
        elif partner_act == 'ask-opinion'
            tag = 'inform-opinion'
        else:
            tag = 'inform-other'
        self.state['movie_informed'][title].append(tag)
        utterance = self.choose_template(tag=tag, movie_title=title.title())['template']
        # TODO: If we couldn't find an utterance with a tag, grab another
        # template from the same movie.
        # if utterance is None:
        #    get other template
        self.state['my_act'] = (tag, None)
        return self.message(utterance)

    def agree(self):
        utterance = self.choose_template(tag='agree')['template']
        self.state['my_act'] = ('agree', None)
        # Agree means we are closing the current thread
        self.state['curr_movie'] = None
        return self.message(utterance)

    def has_unknown_entity(self, entities):
        for entity in entities:
            if entity.canonical.type == 'unknown' or \
                    entity.canonical.value not in self.known_movies:
                return True
        return False

    def bye(self):
        self.state['my_act'] = ('bye', None)
        utterance = self.choose_template('bye')['template']
        return self.message(utterance)

    def send(self):
        print 'SEND'
        partner_act, partner_entities = self.state['partner_act']

        if partner_act == 'done' or self.state['my_act'][0] == 'bye':
            return self.done()

        if partner_act == 'bye':
            return self.bye()

        if partner_act in ('<start>', 'greet'):
            return self.ask(context_tag=partner_act)

        if len(partner_entities) > 0:
            if self.has_unknown_entity(partner_entities):
                return self.ask(context_tag='unknown_entity')
            else:
                assert self.state['curr_movie'] is not None
                return self.inform(title=self.state['curr_movie'])
        elif self.state['curr_movie'] is None:
            return self.start_movie()
        # Partner didn't mention entities but there is a curr_movie
        else:
            my_prev_act, _ = self.state['my_act']
            if partner_act == 'ask' and (not my_prev_act.startswith('start')) and (not my_prev_act.startswith('inform')):
                return self.start_movie()
            elif not self.state['curr_movie'] in self.known_movies:
                return self.agree()
            elif len(self.state['movie_informed'][self.state['curr_movie']]) > 1:
                return self.start_movie()
            else:
                return self.inform(title=self.state['curr_movie'])

        raise Exception('Uncaught case')
