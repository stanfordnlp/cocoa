from collections import defaultdict
import json
import utils
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
from src.basic.negotiation.price_tracker import PriceTracker, PriceScaler, add_price_tracker_arguments
from src.basic.negotiation.tokenizer import tokenize
from src.model.negotiation.preprocess import Preprocessor
from src.basic.scenario_db import NegotiationScenario
from src.basic.entity import Entity
from src.basic.dataset import Example
import nltk.data
import re

__author__ = 'anushabala'


THRESHOLD = 50.0


class SpeechActs(object):
    GEN_QUESTION = 'general_question'
    PRICE_QUESTION = 'price_request'
    GEN_STATEMENT = 'general_statement'
    PRICE_STATEMENT = 'price_statement'
    GREETING = 'greeting'
    AGREEMENT = 'agreement'


class SpeechActAnalyzer(object):
    agreement_patterns = [
        r'that works',
        r'i could do',
        r'[^a-zA-Z]ok[^a-zA-Z]|okay',
        r'great'
    ]
    @classmethod
    def is_question(cls, tokens):
        last_word = tokens[-1]
        first_word = tokens[0]
        return last_word == '?' or first_word in ('how', 'do', 'does', 'are', 'is', 'what', 'would', 'will')

    @classmethod
    def get_question_type(cls, tokens):
        if not cls.is_question(tokens):
            return None
        return cls.is_price_statement(tokens)

    @classmethod
    def is_agreement(cls, raw_sentence):
        for pattern in cls.agreement_patterns:
            if re.match(pattern, raw_sentence, re.IGNORECASE) is not None:
                return True
        return False

    @classmethod
    def is_price_statement(cls, tokens):
        for token in tokens:
            if isinstance(token, Entity) and token.type == 'price':
                return True
            else:
                return False

    @classmethod
    def is_greeting(cls, tokens):
        for token in tokens:
            if token in ('hi', 'hello', 'hey', 'hiya', 'howdy'):
                return True
        return False

    @classmethod
    def get_speech_act(cls, sentence, linked_tokens):
        if cls.is_question(linked_tokens):
            return cls.get_question_type(linked_tokens)
        if cls.is_price_statement(linked_tokens):
            return SpeechActs.PRICE_STATEMENT
        if cls.is_agreement(sentence):
            return SpeechActs.AGREEMENT
        if cls.is_greeting(linked_tokens):
            return SpeechActs.GREETING

        return SpeechActs.GEN_STATEMENT


class StrategyAnalyzer(object):
    def __init__(self, transcripts_path, stats_path, price_tracker_model, debug=False):
        transcripts = json.load(open(transcripts_path, 'r'))
        if debug:
            transcripts = transcripts[:50]
        self.dataset = self.filter_rejected_chats(transcripts)

        self.nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.price_tracker = PriceTracker(price_tracker_model)

        # group chats depending on whether the seller or the buyer wins
        self.buyer_wins, self.seller_wins = self.group_outcomes_and_roles()

        self.stats_path = stats_path
        if not os.path.exists(self.stats_path):
            os.makedirs(self.stats_path)

    @staticmethod
    def filter_rejected_chats(transcripts):
        examples = []
        for chat in transcripts:
            ex = Example.from_dict(None, chat)
            if not Preprocessor.skip_example(ex):
                examples.append(chat)
        return examples

    def group_outcomes_and_roles(self):
        buyer_wins = []
        seller_wins = []
        ties = 0
        total_chats = 0
        for ex in self.dataset:
            roles = {0: ex["scenario"]["kbs"][0]["personal"]["Role"],
                     1: ex["scenario"]["kbs"][1]["personal"]["Role"]}
            winner = utils.get_winner(ex)
            if winner is None:
                continue
            total_chats += 1
            if winner == -1:
                buyer_wins.append(ex)
                seller_wins.append(ex)
                ties += 1
            elif roles[winner] == utils.BUYER:
                buyer_wins.append(ex)
            elif roles[winner] == utils.SELLER:
                seller_wins.append(ex)

        print "# of ties: {:d}".format(ties)
        print "Total chats with outcomes: {:d}".format(total_chats)
        return buyer_wins, seller_wins

    def plot_length_vs_margin(self, out_name='turns_vs_margin.png'):
        labels = ['buyer wins', 'seller wins']
        plt.figure(figsize=(10, 6))

        for (chats, lbl) in zip([self.buyer_wins, self.seller_wins], labels):
            margins = defaultdict(list)
            for ex in chats:
                turns = utils.get_turns_per_agent(ex)
                total_turns = turns[0] + turns[1]
                margin = utils.get_win_margin(ex)
                if margin > 2.5 or margin < 0.:
                    continue

                margins[total_turns].append(margin)

            sorted_keys = list(sorted(margins.keys()))

            turns = []
            means = []
            errors = []
            for k in sorted_keys:
                if len(margins[k]) >= THRESHOLD:
                    turns.append(k)
                    means.append(np.mean(margins[k]))
                    errors.append(stats.sem(margins[k]))

            plt.errorbar(turns, means, yerr=errors, label=lbl, fmt='--o')

        plt.legend()
        plt.xlabel('# of turns in dialogue')
        plt.ylabel('Margin of victory')

        save_path = os.path.join(self.stats_path, out_name)
        plt.savefig(save_path)

    def plot_length_histograms(self):
        lengths = []
        for ex in self.dataset:
            winner = utils.get_winner(ex)
            if winner is None:
                continue
            turns = utils.get_turns_per_agent(ex)
            total_turns = turns[0] + turns[1]
            lengths.append(total_turns)

        hist, bins = np.histogram(lengths)

        width = np.diff(bins)
        center = (bins[:-1] + bins[1:]) / 2

        fig, ax = plt.subplots(figsize=(8,3))
        ax.bar(center, hist, align='center', width=width)
        ax.set_xticks(bins)

        save_path = os.path.join(self.stats_path, 'turns_histogram.png')
        plt.savefig(save_path)

    def _get_price_trend(self, chat, agent=None):
        def _normalize_price(seen_price):
            return (float(seller_target) - float(seen_price)) / (float(seller_target) - float(buyer_target))

        scenario = NegotiationScenario.from_dict(None, chat['scenario'])
        # print chat['scenario']
        kbs = scenario.kbs
        roles = {
            kbs[0].facts['personal']['Role']: 0,
            kbs[1].facts['personal']['Role']: 1
        }

        buyer_target = kbs[roles[utils.BUYER]].facts['personal']['Target']
        seller_target = kbs[roles[utils.SELLER]].facts['personal']['Target']

        prices = []
        for e in chat['events']:
            if e['action'] == 'message':
                if agent is not None and e['agent'] != agent:
                    continue
                raw_tokens = tokenize(e['data'])
                # link entity
                linked_tokens = self.price_tracker.link_entity(raw_tokens,
                                                               kb=kbs[e['agent']])
                for token in linked_tokens:
                    if isinstance(token, Entity):
                        try:
                            replaced = PriceScaler.unscale_price(kbs[e['agent']], token)
                        except OverflowError:
                            print "Raw tokens: ", raw_tokens
                            print "Overflow error: {:s}".format(token)
                            print kbs[e['agent']].facts
                            print "-------"
                            continue
                        norm_price = _normalize_price(replaced.canonical.value)
                        if 0. <= norm_price <= 2.:
                            # if the number is greater than the list price or significantly lower than the buyer's
                            # target it's probably not a price
                            prices.append(norm_price)
                # do some stuff here
            elif e['action'] == 'offer':
                norm_price = _normalize_price(e['data']['price'])
                if 0. <= norm_price <= 2.:
                    prices.append(norm_price)
                # prices.append(e['data']['price'])

        # print "Chat: {:s}".format(chat['uuid'])
        # print "Trend:", prices

        return prices

    def plot_price_trends(self, top_n=10):
        labels = ['buyer_wins', 'seller_wins']
        for (group, lbl) in zip([self.buyer_wins, self.seller_wins], labels):
            plt.figure(figsize=(10, 6))
            trends = []
            for chat in group:
                winner = utils.get_winner(chat)
                margin = utils.get_win_margin(chat)
                if margin > 1.0 or margin < 0.:
                    continue
                if winner is None:
                    continue

                # print "Winner: Agent {:d}\tWin margin: {:.2f}".format(winner, margin)
                if winner == -1 or winner == 0:
                    trend = self._get_price_trend(chat, agent=0)
                    if len(trend) > 1:
                        trends.append((margin, chat, trend))
                if winner == -1 or winner == 1:
                    trend = self._get_price_trend(chat, agent=1)
                    if len(trend) > 1:
                        trends.append((margin, chat,  trend))

                # print ""

            sorted_trends = sorted(trends, key=lambda x:x[0], reverse=True)
            for (idx, (margin, chat, trend)) in enumerate(sorted_trends[:top_n]):
                print '{:s}: Chat {:s}\tMargin: {:.2f}'.format(lbl, chat['uuid'], margin)
                print 'Trend: ', trend
                print chat['scenario']['kbs']
                print ""
                plt.plot(trend, label='Margin={:.2f}'.format(margin))
            plt.legend()
            plt.xlabel('N-th price mentioned in chat')
            plt.ylabel('Value of mentioned price')
            out_path = os.path.join(self.stats_path, '{:s}_trend.png'.format(lbl))
            plt.savefig(out_path)

    def _get_price_mentions(self, chat, agent=None):
        scenario = NegotiationScenario.from_dict(None, chat['scenario'])
        # print chat['scenario']
        kbs = scenario.kbs

        prices = 0
        for e in chat['events']:
            if agent is not None and e['agent'] != agent:
                    continue
            if e['action'] == 'message':
                raw_tokens = tokenize(e['data'])
                # link entity
                linked_tokens = self.price_tracker.link_entity(raw_tokens,
                                                               kb=kbs[e['agent']])
                for token in linked_tokens:
                    if isinstance(token, Entity) and token.type == 'price':
                        prices += 1

        return prices

    def split_turn(self, turn):
        # a single turn can be comprised of multiple sentences
        return self.nltk_tokenizer.tokenize(turn)

    def get_speech_acts(self, chat, agent=None):
        for e in chat['events']:
            
    def plot_speech_acts(self):
        labels = ['buyer_wins', 'seller_wins']
        for (group, lbl) in zip([self.buyer_wins, self.seller_wins], labels):
            plt.figure(figsize=(10, 6))
            mentions = []
            for chat in group:
                winner = utils.get_winner(chat)
                margin = utils.get_win_margin(chat)
                if margin > 1.0 or margin < 0.:
                    continue
                if winner is None:
                    continue

                if winner == -1 or winner == 0:
                    num_mentions = self._get_price_mentions(chat, agent=0)
                    mentions.append((margin, chat, num_mentions))
                if winner == -1 or winner == 1:
                    num_mentions = self._get_price_mentions(chat, agent=1)
                    if len(trend) > 1:
                        mentions.append((margin, chat,  trend))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output-dir', required=True, help='Directory containing all output from website')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (only run on 50 chats)')
    add_price_tracker_arguments(parser)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        raise ValueError("Output directory {:s} doesn't exist".format(args.output_dir))

    transcripts_path = os.path.join(args.output_dir, 'transcripts', 'transcripts.json')
    stats_output = os.path.join(args.output_dir, 'stats')

    analyzer = StrategyAnalyzer(transcripts_path, stats_output, args.price_tracker_model, args.debug)
    # analyzer.plot_length_histograms()
    # analyzer.plot_length_vs_margin()
    analyzer.plot_price_trends()
