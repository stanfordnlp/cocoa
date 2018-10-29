from collections import namedtuple
import re

SpeechAct = namedtuple('SpeechAct', ['name', 'abrv'])

class SpeechActs(object):
    QUESTION = SpeechAct('question', '?')
    PRICE = SpeechAct('price', 'P')
    STATEMENT = SpeechAct('statement', 'S')
    REPEAT_PRICE = SpeechAct('repeat_price', 'P2')
    REFERENCE = SpeechAct('reference', 'R')
    SIDE_OFFER = SpeechAct('side_offer', 'side')
    GREETING = SpeechAct('greeting', 'hi')
    AGREEMENT = SpeechAct('agreement', 'ok')
    PERSON = SpeechAct('person', 'person')
    COND = 'condition'
    POS = 'positive'
    NEG = 'negative'

    #ACT = [GEN_QUESTION, PRICE_QUESTION, GEN_STATEMENT, PRICE_STATEMENT, GREETING, AGREEMENT, PERSON, COND, POS, NEG, REPEAT_PRICE, COUNTER_ARGUMENT]


class SpeechActAnalyzer(object):
    agreement_patterns = [
        r'^that works[.!]*$',
        r'^great[.!]*$',
        r'^(ok|okay)[.!]*$',
        r'^great, thanks[.!]*$',
        r'^deal[.!]*$',
        r'^[\w ]*have a deal[\w ]*$',
        r'^i can do that[.]*$',
    ]

    price_patterns = [
            r'come down',
            r'(highest|lowest)',
            r'go (lower|higher)',
            r'too (high|low)',
            ]

    pos_patterns = [
            r'i can',
            r'i could',
            r'i will',
            r'willing to',
            r'would like to',
            ]

    neg_patterns = [
            r'i cannot',
            r"i couldn't",
            r"i can't",
            r"i won't",
            r"i wouldn't",
            r"i would not",
            ]

    side_offer_patterns = [
            r'deliver|cash|throw in',
            r'pick[ \w]*up',
            ]

    greeting_patterns = [
            r'how are you',
            r'interested in',
            ]

    greeting_words = set(['hi', 'hello', 'hey', 'hiya', 'howdy'])

    question_words = set(['what', 'when', 'where', 'why', 'which', 'who', 'whose', 'how', 'do', 'does', 'are', 'is', 'would', 'will', 'can', 'could'])

    person_words = set(['husband', 'wife', 'son', 'daughter', 'grandma', 'grandmother', 'grandpa', 'grandfarther', 'kid', 'mom', 'dad', 'mother', 'father', 'uncle', 'aunt', 'friend', 'she', 'he', 'brother', 'sister'])

    @classmethod
    def is_question(cls, utterance):
        tokens = utterance.tokens
        if len(tokens) < 1:
            return False
        last_word = tokens[-1]
        first_word = tokens[0]
        return last_word == '?' or first_word in cls.question_words

    @classmethod
    def is_price(cls, utterance):
        for pattern in cls.price_patterns:
            if re.search(pattern, utterance.text):
                return True
        return False

    @classmethod
    def has_person(cls, utterance):
        tokens = utterance.tokens
        for token in tokens:
            if token in cls.person_words:
                return True
        return False

    @ classmethod
    def condition(cls, tokens):
        if 'if' in tokens:
            return True
        return False

    @classmethod
    def sentiment(cls, utterance):
        for pattern in cls.pos_patterns:
            if re.match(pattern, utterance.text, re.IGNORECASE) is not None:
                return 1
        for pattern in cls.neg_patterns:
            if re.match(pattern, utterance.text, re.IGNORECASE) is not None:
                return -1
        return 0

        for token in utterance.tokens:
            if token in ("n't", 'cannot'):
                return -1
            elif token in ('can', 'willing', 'cound'):
                return 1
        return 0

    @classmethod
    def is_agreement(cls, utterance):
        for pattern in cls.agreement_patterns:
            if re.match(pattern, utterance.text, re.IGNORECASE) is not None:
                return True
        return False

    @classmethod
    def has_side_offer(cls, utterance):
        for pattern in cls.side_offer_patterns:
            if re.search(pattern, utterance.text, re.IGNORECASE):
                return True
        return False

    @classmethod
    def has_price(cls, utterance):
        return len(utterance.prices) > 0

    @classmethod
    def is_greeting(cls, utterance):
        for token in utterance.tokens:
            if token in cls.greeting_words:
                return True
        for pattern in cls.greeting_patterns:
            if re.search(pattern, utterance.text, re.IGNORECASE):
                return True
        return False

    @classmethod
    def get_speech_acts(cls, utterance, prev_turn=None):
        acts = []
        if utterance.action != 'message':
            acts.append((SpeechAct(utterance.action, utterance.action), None))
            return acts

        if cls.is_question(utterance):
            acts.append((SpeechActs.QUESTION, None))

        if cls.has_price(utterance):
            acts.append((SpeechActs.PRICE, None))

        if cls.has_side_offer(utterance):
            acts.append((SpeechActs.SIDE_OFFER, None))
        #else:
        #    sentiment = cls.sentiment(text, linked_tokens)
        #    if sentiment == 1:
        #        acts.append(SpeechActs.POS)
        #    elif sentiment == -1:
        #        acts.append(SpeechActs.NEG)

        #if cls.is_agreement(text):
        #    acts.append(SpeechActs.AGREEMENT)

        if cls.is_greeting(utterance):
            acts.append((SpeechActs.GREETING, None))

        if cls.has_person(utterance):
            acts.append((SpeechActs.PERSON, None))

        #if cls.condition(linked_tokens):
        #    acts.append(SpeechActs.COND)

        if prev_turn is not None:
            repeat_prices = cls.get_repeated_prices(utterance, prev_turn)
            repeat_words = cls.get_repeated_keywords(utterance, prev_turn)
            if len(repeat_prices) > 0:
                acts.append((SpeechActs.REPEAT_PRICE, tuple(repeat_prices)))
            if len(repeat_words) > 0:
                acts.append((SpeechActs.REFERENCE, tuple(repeat_words)))

        if len(acts) == 0:
            acts.append((SpeechActs.STATEMENT, None))

        return acts

    @classmethod
    def get_repeated_keywords(cls, utterance, prev_turn):
        repeated = set()
        for prev_word in prev_turn.iter_keywords():
            for word in utterance.keywords:
                if word == prev_word:
                    repeated.add(word)
        return repeated

    @classmethod
    def get_repeated_prices(cls, utterance, prev_turn):
        repeated = set()
        for prev_price in prev_turn.iter_prices():
            for price in utterance.prices:
                if prev_price.canonical == price.canonical:
                    repeated.add(price)
        return repeated

