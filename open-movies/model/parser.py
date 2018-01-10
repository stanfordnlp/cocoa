import re
import copy
import numpy as np
import json

from cocoa.core.entity import is_entity
from cocoa.model.parser import Parser as BaseParser, Utterance
from cocoa.model.parser import LogicalForm as LF

from core.tokenizer import tokenize

class Parser(BaseParser):
    def is_bye(self, utterance):
        s = utterance.text
        if 'bye' in utterance.tokens or re.search(r'(great|nice|fun) (talking|chatting)', s):
            return True
        return False

    #def is_favorite(self, utterance, score=0):
    #    lowered = utterance.text.lower()
    #    if re.search(r'i (just\s|recently\s)?(saw|watched)', lowered):
    #        score += 1.5
    #    if re.search(r'i (really\s)?(prefer|like|love|thought)', lowered):
    #        score += 1
    #    if re.search(r'(i have seen)|(i am planning)', lowered):
    #        score += 1
    #    if re.search(r'(my favorite(s)?)|(the best)', lowered):
    #        score += 1
    #    return score

    #@classmethod
    #def is_generic(cls, utterance, topic):
    #    # if the utterance has an associated movie or genre, it isn't generic
    #    if topic != 'unknown':
    #        return False
    #    if utterance.length > 14:
    #        return False
    #    if re.search(r'\sit|(one is)', utterance.text.lower()):
    #        return True
    #    return False

    #def plot_category(self, utterance, score=0):
    #    if re.search( r'(films?|flicks?|dramas?|movies?|tales?|it|that)\sabout', utterance.text):
    #        score += 1.5
    #    if re.search( r'(stor(y|ies)|documentar(y|ies))\sabout', utterance.text):
    #        score += 1.5
    #    if re.search( r'(story|tale)\sof', utterance.text):
    #        score += 1
    #    if re.search( r'about\sa\s', utterance.text):
    #        score += 1
    #    return score

    # Opinion - What movie do you like?  What is your favorite movie?
    #def opinion_category(self, utterance, score=0):
    #    if re.search(r'movie(s)?|film(s)?', utterance.text.lower()):
    #        score += 0.5
    #    if utterance.tokens[0] == 'what':
    #        score += 0.4
    #    if re.search(r'(did|do) you', utterance.text.lower()):
    #        score += 1
    #    return score

    # Title - Did you like movie X?  My favorite movie is X.
    #def title_category(self, utterance, score=0):
    #    for token in utterance.tokens:
    #        if is_entity(token):
    #            category = token[1][1]
    #            if category == "title":
    #                score += 2
    #    return score

    #def genre_category(self, utterance, score=0):
    #    for token in utterance.tokens:
    #        if is_entity(token):
    #            category = token[1][1]
    #            if category == "genre":
    #                score += 1
    #    if re.search(r'(kind|type) of (movie(s)?|film(s)?)', utterance.text.lower()):
    #        score += 1.1
    #    return score

    #def actor_category(self, utterance, score=0):
    #    for token in utterance.tokens:
    #        if is_entity(token):
    #            category = token[1][1]
    #            if category == "actor":
    #                score += 1.5
    #        if token in ['his', 'he', 'actor']:
    #            score += 0.5
    #    return score

    #def actress_category(self, utterance, score=0):
    #    for token in utterance.tokens:
    #        if is_entity(token):
    #            category = token[1][1]
    #            if category == "actress":
    #                score += 1.5
    #        if token in ['her', 'she', 'actress']:
    #            score += 0.5
    #    return score

    # see manager.py for more information
    #def balance_evidence(self, intent_scores, topic_scores):
    #    intents = ['greet', 'inquire', 'favorite']
    #    intent_sum = np.sum(intent_scores)
    #    intent_idx = np.argmax(intent_scores)
    #    intent = intents[intent_idx] if intent_sum > 0.5 else "unknown"

    #    topics = ['plot', 'title', 'genre', 'actor', 'actress', 'opinion']
    #    topic_sum = np.sum(topic_scores)
    #    topic_idx =  np.argmax(topic_scores)
    #    topic = topics[topic_idx] if topic_sum > 0.5 else None

    #    if topic in ["actor", "actress", "title"] and (topic_sum > 1) and (intent == "unknown"):
    #        intent = "inquire" if intent_scores[1] > intent_scores[2] else "favorite"
    #    if topic in ["plot", "opinion"] and (topic_sum >= 2) and (intent == "unknown"):
    #        intent = "inquire"
    #    if (intent == "greet") and (topic is not None) and (intent_sum - topic_sum > 1):
    #        topic = None
    #    elif (intent == "inquire"):
    #        if (topic is None) and (topic_sum == 0.5) and (intent_sum > 1):
    #            topic = topics[topic_idx]
    #        if topic == "opinion" and (topic_scores[2] > 1):
    #            topic = "genre"
    #    elif (intent == "favorite") and (topic is None) and (intent_sum > 1):
    #        topic = "title"

    #    # print("intent scores: {0} is {1}".format(intent_scores, intent) )
    #    # print("topic scores: {0} is {1}".format(topic_scores, topic) )

    #    if LogicalForm.is_valid_label(intent, topic):
    #        return LogicalForm(intent, topic)
    #    else:
    #        return LogicalForm("unknown")

    def get_entities(self, tokens, type_=None):
        entities = [x for x in tokens if is_entity(x) and (type_ is None or x.canonical.type == type_)]
        return entities

    def extract_template(self, tokens, dialogue_state):
        return [x.surface if is_entity(x) else x for x in tokens]

    def classify_question(self, utterance):
        if utterance.text.endswith('about?'):
            return 'ask-plot'
        if 'genre' in utterance.tokens:
            return 'ask-genre'
        for token in utterance.tokens:
            if token in ('movie', 'movies', 'favorite', 'seen', 'watch'):
                return 'ask-movie'
        return 'ask'

    def classify_intent(self, utterance, dialogue_state):
        titles = [x.canonical.value for x in self.get_entities(utterance.tokens, 'title')]
        if titles:
            if dialogue_state.curr_title in titles:
                intent = 'inform-curr-title'
            else:
                intent = 'inform-new-title'
        elif self.get_entities(utterance.tokens, 'person'):
            intent = 'inform-person'
        elif dialogue_state.time == 0:
            intent = 'intro'
        elif self.is_question(utterance):
            intent = self.classify_question(utterance)
        elif self.is_greeting(utterance):
            intent = 'greet'
        elif self.is_bye(utterance):
            intent = 'bye'
        elif dialogue_state.partner_act.startswith('ask'):
            intent = 'inform'
        else:
            intent = 'unknown'
        return intent

    #def deduce_topic(self, utterance):
    #    plot_score = self.plot_category(utterance)
    #    title_score = self.title_category(utterance)
    #    genre_score = self.genre_category(utterance)
    #    actor_score = self.actor_category(utterance)
    #    actress_score = self.actress_category(utterance)
    #    opinion_score = self.opinion_category(utterance)
    #    return np.array([plot_score, title_score, genre_score, actor_score,
                                        #actress_score, opinion_score])

    def parse(self, event, dialogue_state):
        # We are parsing the partner's utterance
        assert event.agent == 1 - self.agent
        if event.action == 'done':
            u = self.parse_action(event)
        elif event.action == 'message':
            u = self.parse_message(event, dialogue_state)
        else:
            return False

        return u

    def parse_message(self, event, dialogue_state):
        tokens = self.lexicon.link_entity(event.data)
        tokens = [x.lower() if not is_entity(x) else x for x in tokens]
        utterance = Utterance(raw_text=event.data, tokens=tokens)
        intent = self.classify_intent(utterance, dialogue_state)
        template = self.extract_template(tokens, dialogue_state)
        utterance.lf = LF(intent, titles=self.get_entities(tokens, 'title'))
        utterance.template = template
        return utterance
