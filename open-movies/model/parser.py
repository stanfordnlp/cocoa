import re
import copy
import numpy as np
import json

from cocoa.core.entity import is_entity
from cocoa.model.parser import Parser as BaseParser, Utterance
from cocoa.model.parser import LogicalForm as BaseLF
from nltk import sent_tokenize, word_tokenize

class LogicalForm(BaseLF):
    def __init__(self, intent, topic=None):
        self.intent = intent
        self.topic = topic
        suffix = "-" + topic if topic else ""
        self.full_intent = intent + suffix

    @staticmethod
    def is_valid_label(intent, topic):
        valid_labels = ["greet", "inquire", "inquire-title", "inquire-genre",
            "inquire-plot", "inquire-opinion", "inquire-actor", "inquire-actress",
            "favorite-actor","favorite-actress","favorite-genre","favorite-title",
            "tellmore-actor","tellmore-actress","tellmore-genre","tellmore-title"]
        suffix = "-" + topic if topic else ""
        full_intent = intent + suffix
        return full_intent in valid_labels

    def to_dict(self):
        attrs['intent'] = self.intent
        attrs['topic'] = self.topic
        return attrs

class Parser(BaseParser):
    question_words = set(['what', 'when', 'where', 'why', 'which', 'who',
            'whose', 'how', 'do', 'did', 'have'])

    def is_greeting(self, utterance, score=0):
        if super(Parser, self).is_greeting(utterance):
            score += 2
        if utterance.length < 7:
            score += 0.5
        return score

    def is_question(self, utterance, score=0):
        if utterance.length < 1:
            return score
        for idx, token in enumerate(utterance.tokens):
            if is_entity(token): continue
            if token.lower() in question_words:
                score += 0.5
                if idx == 0:
                    score += 0.5
            if token == '?':
                score += 1
        return score

    def is_favorite(self, utterance, score=0):
        lowered = utterance.text.lower()
        if re.search(r'i (just\s|recently\s)?(saw|watched)', lowered):
            score += 1.5
        if re.search(r'i (really\s)?(prefer|like|love|thought)', lowered):
            score += 1
        if re.search(r'(i have seen)|(i am planning)', lowered):
            score += 1
        if re.search(r'(my favorite(s)?)|(the best)', lowered):
            score += 1
        return score

    @classmethod
    def is_generic(cls, utterance, topic):
        # if the utterance has an associated movie or genre, it isn't generic
        if topic != 'unknown':
            return False
        if utterance.length > 14:
            return False
        if re.search(r'\sit|(one is)', utterance.text.lower()):
            return True
        return False

    def extract_template(self, tokens, dialogue_state):
        template = []
        for token in tokens:
            if self._is_item(token):
                item = token.canonical.value
                if template and template[-1] == '{number}':
                    template[-1] = '{{{0}-number}}'.format(item)
                template.append('{{{0}}}'.format(item))
            elif self._is_number(token):
                template.append('{number}')
            else:
                template.append(token)
        return template

    def plot_category(self, utterance, score=0):
        if re.search( r'(films?|flicks?|dramas?|movies?|tales?|it|that)\sabout', utterance.text):
            score += 1.5
        if re.search( r'(stor(y|ies)|documentar(y|ies))\sabout', utterance.text):
            score += 1.5
        if re.search( r'(story|tale)\sof', utterance.text):
            score += 1
        if re.search( r'about\sa\s', utterance.text):
            score += 1
        return score

    # Opinion - What movie do you like?  What is your favorite movie?
    def opinion_category(self, utterance, score=0):
        if re.search(r'movie(s)?|film(s)?', utterance.text.lower()):
            score += 0.5
        if utterance.tokens[0] == 'what':
            score += 0.4
        if re.search(r'(did|do) you', utterance.text.lower()):
            score += 1
        return score

    # Title - Did you like movie X?  My favorite movie is X.
    def title_category(self, utterance, score=0):
        for token in utterance.tokens:
            if is_entity(token):
                category = token[1][1]
                if category == "title":
                    score += 2
        return score

    def genre_category(self, utterance, score=0):
        for token in utterance.tokens:
            if is_entity(token):
                category = token[1][1]
                if category == "genre":
                    score += 1
        if re.search(r'(kind|type) of (movie(s)?|film(s)?)', utterance.text.lower()):
            score += 1.1
        return score

    def actor_category(self, utterance, score=0):
        for token in utterance.tokens:
            if is_entity(token):
                category = token[1][1]
                if category == "actor":
                    score += 1.5
            if token in ['his', 'he', 'actor']:
                score += 0.5
        return score

    def actress_category(self, utterance, score=0):
        for token in utterance.tokens:
            if is_entity(token):
                category = token[1][1]
                if category == "actress":
                    score += 1.5
            if token in ['her', 'she', 'actress']:
                score += 0.5
        return score

    # see manager.py for more information
    def balance_evidence(self, intent_scores, topic_scores):
        intents = ['greet', 'inquire', 'favorite']
        intent_sum = np.sum(intent_scores)
        intent_idx = np.argmax(intent_scores)
        intent = intents[intent_idx] if intent_sum > 0.5 else "unknown"

        topics = ['plot', 'title', 'genre', 'actor', 'actress', 'opinion']
        topic_sum = np.sum(topic_scores)
        topic_idx =  np.argmax(topic_scores)
        topic = topics[topic_idx] if topic_sum > 0.5 else None

        if topic in ["actor", "actress", "title"] and (topic_sum > 1) and (intent == "unknown"):
            intent = "inquire" if intent_scores[1] > intent_scores[2] else "favorite"
        if topic in ["plot", "opinion"] and (topic_sum >= 2) and (intent == "unknown"):
            intent = "inquire"
        if (intent == "greet") and (topic is not None) and (intent_sum - topic_sum > 1):
            topic = None
        elif (intent == "inquire"):
            if (topic is None) and (topic_sum == 0.5) and (intent_sum > 1):
                topic = topics[topic_idx]
            if topic == "opinion" and (topic_scores[2] > 1):
                topic = "genre"
        elif (intent == "favorite") and (topic is None) and (intent_sum > 1):
            topic = "title"

        print("intent scores: {0} is {1}".format(intent_scores, intent) )
        print("topic scores: {0} is {1}".format(topic_scores, topic) )

        if LogicalForm.is_valid_label(intent, topic):
            return LogicalForm(intent, topic)
        else:
            return LogicalForm("unknown")

    def classify_intent(self, utterance):
        greet_score = self.is_greeting(utterance)
        inquire_score = self.is_question(utterance)
        favorite_score = self.is_favorite(utterance)
        return np.array([greet_score, inquire_score, favorite_score])

    def deduce_topic(self, utterance):
        plot_score = self.plot_category(utterance)
        title_score = self.title_category(utterance)
        genre_score = self.genre_category(utterance)
        actor_score = self.actor_category(utterance)
        actress_score = self.actress_category(utterance)
        opinion_score = self.opinion_category(utterance)
        return np.array([plot_score, title_score, genre_score, actor_score,
                                        actress_score, opinion_score])

    def parse_message(self, raw_utterance):
        entities = self.lexicon.link_entity(raw_utterance)
        utterance = Utterance(raw_text=raw_utterance, tokens=entities)
        intent_scores = self.classify_intent(utterance)
        topic_scores = self.deduce_topic(utterance)
        utterance.lf = self.balance_evidence(intent_scores, topic_scores)

        return utterance

    def unit_test(self, examples, verbose=False):
        correct = 0
        for exp in examples:
            x, y = exp[0], exp[1]
            utterance = self.parse_message(x)
            intent = utterance.lf.full_intent
            if (intent == y):
                correct += 1
            if verbose:
                if (intent == y):
                    print("Correct")
                else:
                    print utterance.tokens
                    print("Message: {}".format(x) )
                    print("Predicted Label: {}".format(intent) )
                    print("Actual Label: {}".format(y) )
                print("------------------------------")
        print("Passed {} out of {} examples".format(correct, len(examples)) )

if __name__ == '__main__':
    import argparse
    from core.kb import KB
    from core.lexicon import Lexicon
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lexicon', help='Path to pickled lexicon')
    parser.add_argument('-e', '--examples-path', help='Path to training json file')
    parser.add_argument('--verbose', default=False, action='store_true', help='Debug mode')
    args = parser.parse_args()
    # PYTHONPATH=. python model/parser.py -l data/lexicon.pkl --examples-path data/parser_unit.json --verbose

    lexicon = Lexicon.from_pickle(args.lexicon)
    movie_parser = Parser(0, {}, lexicon)
    examples = json.load(open(args.examples_path, "r"))
    movie_parser.unit_test(examples, args.verbose)