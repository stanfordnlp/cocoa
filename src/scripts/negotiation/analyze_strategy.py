from collections import defaultdict
import json

__author__ = 'anushabala'
import utils
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
from src.basic.price_tracker import PriceTracker
from src.model.preprocess import tokenize


THRESHOLD = 50.0


class StrategyAnalyzer(object):
    def __init__(self, transcripts_path, stats_path):
        self.transcripts = json.load(open(transcripts_path, 'r'))

        self.price_tracker = PriceTracker()
        # group chats depending on whether the seller or the buyer wins
        self.buyer_wins, self.seller_wins = self.group_outcomes_and_roles()

        self.stats_path = stats_path
        if not os.path.exists(self.stats_path):
            os.makedirs(self.stats_path)

    def group_outcomes_and_roles(self):
        buyer_wins = []
        seller_wins = []
        ties = 0
        total_chats = 0
        for t in self.transcripts:
            roles = {0: t["scenario"]["kbs"][0]["personal"]["Role"],
                     1: t["scenario"]["kbs"][1]["personal"]["Role"]}
            winner = utils.get_winner(t)
            if winner is None:
                continue
            total_chats += 1
            if winner == -1:
                buyer_wins.append(t)
                seller_wins.append(t)
                ties += 1
            elif roles[winner] == utils.BUYER:
                buyer_wins.append(t)
            elif roles[winner] == utils.SELLER:
                seller_wins.append(t)

        print "# of ties: {:d}".format(ties)
        print "Total chats with outcomes: {:d}".format(total_chats)
        return buyer_wins, seller_wins

    def plot_length_vs_margin(self, out_name='turns_vs_margin.png'):
        labels = ['buyer wins', 'seller wins']
        plt.figure(figsize=(10, 6))

        for (chats, lbl) in zip([self.buyer_wins, self.seller_wins], labels):
            margins = defaultdict(list)
            for t in chats:
                turns = utils.get_turns_per_agent(t)
                total_turns = turns[0] + turns[1]
                margin = utils.get_win_margin(t)
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
        for t in self.transcripts:
            winner = utils.get_winner(t)
            if winner is None:
                continue
            turns = utils.get_turns_per_agent(t)
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

    def _get_price_trend(self, chat):
        def _normalize_price(seen_price):
            return (seller_target - seen_price) / (seller_target - buyer_target)

        kbs = chat['scenario']['kbs']
        roles = {
            kbs[0]['personal']['Role']: 0,
            kbs[1]['personal']['Role']: 1
        }

        buyer_target = kbs[roles[utils.BUYER]]['personal']['Target']
        seller_target = kbs[roles[utils.SELLER]]['personal']['Target']

        prices = []
        # todo process events separately for seller vs buyer - for each price trend plot, show change in prices
        # mentioned by seller vs change in prices mentioned by buyer
        for e in chat['events']:
            if e['action'] == 'message':
                tokens = tokenize(e['data'])
                # link entity
                linked_tokens = self.price_tracker.link_entity(tokens,
                                                               kb=kbs[e['agent']],
                                                               partner_kb=kbs[1-e['agent']])
                # do some stuff here
            elif e['action'] == 'offer':
                prices.append(_normalize_price(e['data']))

        pass

    def plot_price_trends(self):
        pass

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output-dir', required=True, help='Directory containing all output from website')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        raise ValueError("Output directory {:s} doesn't exist".format(args.output_dir))

    transcripts_path = os.path.join(args.output_dir, 'transcripts', 'transcripts.json')
    stats_output = os.path.join(args.output_dir, 'stats')

    analyzer = StrategyAnalyzer(transcripts_path, stats_output)
    analyzer.plot_length_histograms()
    analyzer.plot_length_vs_margin()
