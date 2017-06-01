from collections import defaultdict
import os

__author__ = 'anushabala'
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt


def json_to_text(transcript, agent_type=None):
    """
    Convert a JSON chat transcript to a single string containing all the messages in the chat.
    :param json_chat:
    :return:
    """
    chat = []
    agents = transcript["agents"]
    for event in transcript["events"]:
        if agent_type is not None and agents[str(event["agent"])] != agent_type:
            continue
        if event["action"] == "message":
            chat.append(event["data"].strip())

    return " ".join(chat)


def plot_top_tokens(top_features_by_agent, agents=["human"], output_dir="./"):
    num_subplots = len(top_features_by_agent)
    categories = top_features_by_agent[0].keys()
    for cat in categories:
        plt.figure(1)
        for i in xrange(0, num_subplots):
            sp = int("{:d}1{:d}".format(num_subplots, i+1))
            plt.subplot(sp)

            top = top_features_by_agent[i][cat]
            tokens = [x[0] for x in top]
            scores = [x[1] for x in top]
            x_vals = [k+1 for k in xrange(len(top))]

            plt.xlabel("n-th most important token")
            plt.ylabel("tf-idf score")
            plt.scatter(x_vals, scores, c='m', marker='o', s=18)
            for (k, txt) in enumerate(tokens):
                plt.text(s="  "+txt, x=x_vals[k]+0.25, y=scores[k], rotation='vertical', ha='left')

            plt.gca().set_ylim(bottom=-0.05)
            plt.title("Agent type: {:s} Category: {:s}".format(agents[i], cat))

        plt.tight_layout()
        outpath = os.path.join(output_dir, "{:s}.png".format(cat))

        plt.savefig(outpath)
        plt.clf()


class TfIdfCalculator(object):
    """
    Computes tf-idf scores between a listing and all chats associated with that listing.
    """

    def __init__(self, transcripts, top_n=5, agent_type=None):
        self.grouped_chats = None
        self.vectorizer = None
        self.listings = set()
        self.categories = set()
        self.tfs = None
        self.top_n = top_n
        self.agent_type = agent_type

        self.build(transcripts)

    def build(self, transcripts):
        grouped_chats = defaultdict(str)
        for t in transcripts:
            chat = json_to_text(t, self.agent_type)
            category = t["scenario"]["category"]
            grouped_chats[category] += " {:s}".format(chat)

            self.categories.add(category)

        # self.listings = list(sorted(self.listings))
        self.categories = list(sorted(self.categories))
        self.grouped_chats = [grouped_chats[cat] for cat in self.categories]

        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfs = self.vectorizer.fit_transform(self.grouped_chats)
        self.tfs = self.tfs.todense()

    def analyze(self):
        feature_names = self.vectorizer.get_feature_names()
        sorted_tfs = np.argsort(self.tfs, axis=1)
        top_features = {}
        for (idx, cat) in enumerate(self.categories):
            top_tfs = sorted_tfs[idx, :-self.top_n-1:-1].tolist()[0]
            top_features[cat] = [(feature_names[col], self.tfs[idx, col]) for col in top_tfs]

        for cat in self.categories:
            if self.agent_type is not None:
                print "Agent type: {:s}\t".format(self.agent_type),
            print "Category {:s}".format(cat)
            for (fname, col) in top_features[cat]:
                print "{:s}: {:.5f}".format(fname, col)
            print "---------------------------------------------------"

        return top_features
