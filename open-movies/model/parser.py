import re
import copy
import numpy as np
import json

from cocoa.core.entity import is_entity
from cocoa.model.parser import Parser as BaseParser, LogicalForm as LF, Utterance
from nltk import sent_tokenize, word_tokenize

class Parser(BaseParser):
    sentence_delimiter = ('.', ';', '?')

    @classmethod
    def is_greeting(cls, utterance):
        greet = super(Parser, cls).is_greeting(utterance)
        if greet and utterance.length < 7:
            return True
        return False

    @classmethod
    def is_favorite(cls, utterance):
        lowered = utterance.text.lower()
        if re.search(r'i (just\s|recently\s)?(saw|watched)', lowered):
            return True
        if re.search(r'i (really\s)?(prefer|like|love)', lowered):
            return True
        if re.search(r'(i have seen)|(i am planning)|favorite(s)?', lowered):
            return True
        return False

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

    # Returns a tuple of confidence score and topic, which is
    # either 'title' or 'opinion', does NOT return the title of the movie
    # Examples:
    #   Opinion - What movie do you like?  What is your favorite movie?
    #   Title - Did you like movie X?  My favorite movie is X.
    def movie_category(utterance):
        score, category = 0, 'opinion'
        if 'movie' in utterance.lf: # logical forms
            score += 3
            category = 'title'
        if re.search(r'movie(s)?', utterance.text.lower()):
            if utterance.tokens[0] == 'what':
                score += 1
            score += 1
        return score, category

    # Returns a tuple of confidence score and the string 'genre',
    # does NOT return the actual name of the genre
    def genre_category(utterance):
        score, category = 0, 'genre'
        for token in utterance.tokens:
            if token in ['genre', 'genres']:
                score += 0.5
            if token in self.genres:
                score += 1
        return score, category

    # Returns a tuple of confidence score and topic, which is
    # either 'actor' or 'actress', does NOT return the name of the actor
    def actor_category(utterance):
        score, category = 0, 'actor'
        for token in utterance.tokens:
            if token in ['his', 'he', 'actor']:
                score += 0.5
            if token in ['her', 'she', 'actress']:
                score += 0.5
                category = "actress"
            if token in self.actors:
                score += 2
            if token in self.actresses:
                score += 2
                category = "actress"
        # if token in NLTK named entity recognizer
        if re.search(r'((her|his) acting)|(s)?he acts', utterance.text.lower()):
            score += 1
        return score, category

    # ['movie', 'opinion', 'genre', 'actor', 'actress', 'unknown']
    def deduce_topic(self, utterance, category):
        movie_score, movie = self.movie_category(utterance)
        genre_score, genre = self.genre_category(utterance)
        actor_score, actor = self.actor_category(utterance)

        topics = [movie, genre, actor]
        scores = np.array([title_score, genre_score, actor_score])
        topic_index = np.argmax(scores)

        return topics[topic_index] if np.sum(scores) > 0.5 else "unknown"

    def classify_intent(self, utterance):
        topic = self.deduce_topic(utterance)
        if self.is_greeting(utterance):
            intent = 'greet'
        elif self.is_question(utterance):
            intent = 'inquire-{}'.format(topic)
        elif self.is_favorite(utterance):
            intent = 'favorite-{}'.format(topic)
        elif self.is_generic(utterance, topic):
            intent = 'generic'
        else:
            intent = 'unknown'
        return intent

    def parse_message(self, raw_utterance):
        intent = "sandwich"
        '''
        entities = self.lexicon.link_entity(tokenize(raw_utterance))
        utterance = Utterance(raw_text=raw_utterance, tokens=entities)
        intent = self.classify_intent(utterance)

        lf = LF(intent, proposal=split, proposal_type=proposal_type)
        utterance.lf = lf
        utterance.template = self.extract_template(tokens, dialogue_state)
        utterance.ambiguous_template = ambiguous_proposal
        '''
        return intent

    def unit_test(self, examples):
        for exp in examples:
            x, y = exp[0], exp[1]
            intent = self.parse_message(x)
            print("Message: {}".format(x) )
            print("Predicted Label: {}".format(intent) )
            print("Actual Label: {}".format(y) )
            print("------------------------------")

if __name__ == '__main__':
    import argparse
    from core.kb import KB
    from core.lexicon import Lexicon
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lexicon', help='Path to pickled lexicon')
    parser.add_argument('-e', '--examples-path', help='Path to training json file')
    args = parser.parse_args()
    # PYTHONPATH=. python model/parser.py -l data/lexicon.pkl --examples-path data/parser_unit.json

    lexicon = Lexicon.from_pickle(args.lexicon)
    movie_parser = Parser(0, None, lexicon)
    movie_parser.unit_test(json.load(open(args.examples_path, "r")))