import os

__author__ = 'anushabala'
from collections import defaultdict
from utils import WINNER, LOSER, get_winner, ROLES, get_category
from cocoa.turk.accept_negotiation_hits import is_chat_valid, is_partial_chat
from cocoa.core.negotiation.tokenizer import tokenize
import matplotlib.pyplot as plt


def group_text_by_winner(transcripts, agent_type=None):
    grouped_chats = defaultdict(list)
    for t in transcripts:
        if (not is_chat_valid(t, 0) and not is_partial_chat(t, 0)) \
                or (not is_chat_valid(t, 1) and not is_partial_chat(t, 1)):
            continue
        winner = get_winner(t)
        if winner is None:
            continue
        utterances_0 = json_to_utterances(t, agent_type, role=None, agent_id=0)
        utterances_1 = json_to_utterances(t, agent_type, role=None, agent_id=1)
        if winner == -1:
            grouped_chats[WINNER].extend(utterances_0)
            grouped_chats[WINNER].extend(utterances_1)
            grouped_chats[LOSER].extend(utterances_0)
            grouped_chats[LOSER].extend(utterances_1)
        elif winner == 0:
            grouped_chats[WINNER].extend(utterances_0)
            grouped_chats[LOSER].extend(utterances_1)
        elif winner == 1:
            grouped_chats[WINNER].extend(utterances_1)
            grouped_chats[LOSER].extend(utterances_0)

    return grouped_chats


def group_text_by_category(transcripts, agent_type=None):
    grouped_chats = defaultdict(list)
    for t in transcripts:
        if (not is_chat_valid(t, 0) and not is_partial_chat(t, 0)) \
                or (not is_chat_valid(t, 1) and not is_partial_chat(t, 1)):
            continue
        utterances = json_to_utterances(t, agent_type)
        category = get_category(t)
        grouped_chats[category].extend(utterances)
    return grouped_chats


def group_text_by_role(transcripts, agent_type=None):
    grouped_chats = defaultdict(list)
    for t in transcripts:
        if (not is_chat_valid(t, 0) and not is_partial_chat(t, 0)) \
                or (not is_chat_valid(t, 1) and not is_partial_chat(t, 1)):
            continue
        for r in ROLES:
            utterances = json_to_utterances(t, agent_type, r)
            grouped_chats[r].extend(utterances)
    return grouped_chats


def json_to_utterances(transcript, agent_type=None, role=None, agent_id=-1):
    """
    Convert a JSON chat transcript to a single string containing all the messages in the chat.
    :param json_chat:
    :return:
    """
    chat = []
    agents = transcript["agents"]
    roles = {0: transcript["scenario"]["kbs"][0]["personal"]["Role"],
             1: transcript["scenario"]["kbs"][1]["personal"]["Role"]}
    for event in transcript["events"]:
        if agent_type is not None and agents[str(event["agent"])] != agent_type:
            continue
        if role is not None and roles[event["agent"]] != role:
            continue
        if agent_id != -1 and agent_id != event["agent"]:
            continue
        if event["action"] == "message":
            chat.append(event["data"].strip())

    return chat


def plot_top_ngrams(top_ngrams_by_agent, agents=["human"], output_dir="./", suffix=""):
    num_subplots = len(top_ngrams_by_agent)
    categories = top_ngrams_by_agent[0].keys()
    for cat in categories:
        plt.figure(1)
        for i in xrange(0, num_subplots):
            sp = int("{:d}1{:d}".format(num_subplots, i+1))
            plt.subplot(sp)

            top = top_ngrams_by_agent[i][cat]
            tokens = [x[0] for x in top]
            counts = [x[1] for x in top]
            x_vals = [k+1 for k in xrange(len(top))]

            plt.xlabel("n-th most important token")
            plt.ylabel("tf-idf score")
            plt.scatter(x_vals, counts, c='m', marker='o', s=18)
            for (k, txt) in enumerate(tokens):
                plt.text(s=k, x=x_vals[k] + 0.25, y=counts[k], ha='left')

            plt.gca().set_ylim(bottom=-0.05)
            plt.title("Agent type: {:s} Category: {:s}".format(agents[i], cat))

        plt.tight_layout()
        outpath = os.path.join(output_dir, "ngram_{:s}{:s}.png".format(cat))

        plt.savefig(outpath)
        plt.clf()


def write_to_file(top_ngrams_by_agent, agents=["human"], output_dir="./", suffix=""):
    for (idx, agent) in enumerate(agents):
        top_ngrams_by_cat = top_ngrams_by_agent[idx]
        for cat in top_ngrams_by_cat.keys():
            outpath = os.path.join(output_dir, "top_ngrams_{:s}_{:s}{:s}.txt".format(agent, cat, suffix))
            outfile = open(outpath, 'w')
            print "N-grams: Agent type: {:s}\tCategory: {:s}".format(agent, cat)
            for (ngram, count) in top_ngrams_by_cat[cat]:
                outfile.write("{:.1f}\t{:s}\n".format(count, " ".join(ngram)))
                print "\t{:.1f}\t{:s}".format(count, " ".join(ngram))


class NgramAnalyzer(object):
    def __init__(self, grouped_utterances, n=5, top_n=20, agent_type=None):
        self.grouped_utterances = grouped_utterances
        self.categories = set(grouped_utterances.keys())
        self.vectorizer = None
        self.listings = set()
        self.tfs = None
        self.n = n
        self.top_n = top_n
        self.ngram_counts = {}
        self.agent_type = agent_type
        self.build()

    def build(self):
        for cat in self.categories:
            self.ngram_counts[cat] = defaultdict(float)
            for utterance in self.grouped_utterances[cat]:
                tokens = tokenize(utterance)
                if len(tokens) < self.n:
                    self.ngram_counts[cat][tuple(tokens)] += 1.
                else:
                    for i in xrange(0, len(tokens) - self.n + 1):
                        ngram = tuple(tokens[i: i + self.n])
                        self.ngram_counts[cat][ngram] += 1.

    def analyze(self):
        top_ngrams = {}
        for cat in self.categories:
            counts = self.ngram_counts[cat]
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            top_ngrams[cat] = sorted_counts[:self.top_n]

        return top_ngrams



