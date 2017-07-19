import json

__author__ = 'anushabala'
import utils
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np


def group_outcomes_and_roles(chats):
    buyer_wins = []
    seller_wins = []
    ties = 0
    chats = 0
    for t in chats:
        roles = {0: t["scenario"]["kbs"][0]["personal"]["Role"],
                 1: t["scenario"]["kbs"][1]["personal"]["Role"]}
        winner = utils.get_winner(t)
        if winner is None:
            continue
        chats += 1
        if winner == -1:
            buyer_wins.append(t)
            seller_wins.append(t)
            ties += 1
        elif roles[winner] == utils.BUYER:
            buyer_wins.append(t)
        elif roles[winner] == utils.SELLER:
            seller_wins.append(t)

    print "# of ties: {:d}".format(ties)
    print "Total chats with outcomes: {:d}".format(chats)
    return buyer_wins, seller_wins


def plot_length_vs_margin(chats, title="Length vs. margin of win"):
    margins = {
        (5, 7.5): [],
        (7.5, 10): [],
        (10., 12.5): []
    }

    for t in chats:
        turns = utils.get_turns_per_agent(t)
        total_turns = turns[0] + turns[1]
        margin = utils.get_win_margin(t)
        # if margin > 1.:
        #     print "Margin > 1: {:.1f}".format(margin)
        if margin > 1.5 or margin < 0.:
            continue
        for (a, b) in margins.keys():
            if a <= total_turns < b:
                margins[(a, b)].append(margin)

    for (a, b) in margins.keys():
        plt.hist(margins[(a, b)], bins='auto', alpha=0.75, label='{:.1f} <= # turns < {:.1f}'.format(a, b))
    plt.xlabel('Margin of victory')
    plt.ylabel('# of chats')
    plt.legend()
    plt.show()


def plot_length_histograms(chats):
    lengths = []
    for t in chats:
        winner = utils.get_winner(t)
        if winner is None:
            continue
        turns = utils.get_turns_per_agent(t)
        total_turns = turns[0] + turns[1]
        lengths.append(total_turns)

    hist, bins = np.histogram(lengths)
    print bins

    width = np.diff(bins)
    center = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(8,3))
    ax.bar(center, hist, align='center', width=width)
    ax.set_xticks(bins)
    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--transcripts', required=True, type=str, help='Path to transcripts JSON file')
    args = parser.parse_args()

    transcripts = json.load(open(args.transcripts, 'r'))
    # group_outcomes_and_roles(transcripts)
    plot_length_vs_margin(transcripts)
    # plot_length_histograms(transcripts)