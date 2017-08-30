from collections import defaultdict
import json
import os
from cocoa.core.entity import Entity
import utils
from analyze_strategy import StrategyAnalyzer, SpeechActs
from cocoa.core.negotiation.price_tracker import PriceTracker, add_price_tracker_arguments
from cocoa.core.negotiation.tokenizer import tokenize
from cocoa.model.negotiation.preprocess import markers as SpecialSymbols
from cocoa.core.scenario_db import NegotiationScenario
from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
import numpy as np

__author__ = 'anushabala'


PRICE = 'PRICE'


class Featurizer(object):
    MIN_NGRAM_THRESHOLD = 20

    def __init__(self, transcripts, price_tracker_model, **kwargs):
        # filter all rejected chats and all incomplete chats
        self.transcripts = utils.filter_incomplete_chats(utils.filter_rejected_chats(transcripts))
        print "Loaded {:d} chats".format(len(self.transcripts))
        self.price_tracker = PriceTracker(price_tracker_model)

        self.first_agent_features = kwargs['first_agent_features'] if 'first_agent_features' in kwargs else True
        self.category_features = kwargs['category_features'] if 'category_features' in kwargs else True
        self.ngram_features = kwargs['ngram_features'] if 'ngram_features' in kwargs else True
        self.turn_features = kwargs['turn_features'] if 'turn_features' in kwargs else True
        self.token_features = kwargs['token_features'] if 'token_features' in kwargs else True
        self.price_features = kwargs['price_features'] if 'price_features' in kwargs else True
        self.speech_act_features = kwargs['speech_act_features'] if 'speech_act_features' in kwargs else True

        if self.category_features:
            self.categories = set()

        if self.ngram_features:
            self.min_n = kwargs['min_n'] if 'min_n' in kwargs else 1
            self.max_n = kwargs['max_n'] if 'max_n' in kwargs else 2
            self.buyer_ngrams = defaultdict(int)
            self.seller_ngrams = defaultdict(int)

        self.feature_names = []

        self.build()

    def build(self):
        # extract all possible categories
        self.extract_categories()
        self.categories = sorted(self.categories)

        # extract all possible n-grams by role
        self.extract_ngram_features(utils.BUYER)
        self.extract_ngram_features(utils.SELLER)

        self.buyer_ngrams = sorted([k for (k, v) in self.buyer_ngrams.items() if v >= self.MIN_NGRAM_THRESHOLD])
        print 'Extracted {:d} buyer n-grams'.format(len(self.buyer_ngrams))
        self.seller_ngrams = sorted([k for (k, v) in self.seller_ngrams.items() if v >= self.MIN_NGRAM_THRESHOLD])
        print 'Extracted {:d} seller n-grams'.format(len(self.seller_ngrams))

        self.build_feature_names()

    def build_feature_names(self):
        self.feature_names.append("first_agent")

        self.feature_names.extend(self.categories)

        self.feature_names.append("turns")

        self.feature_names.append("buyer_tokens")
        self.feature_names.append("seller_tokens")

        self.feature_names.append("buyer_first_price")
        self.feature_names.append("seller_first_price")

        for role in [utils.BUYER, utils.SELLER]:
            for act in SpeechActs.ACTS:
                self.feature_names.append("{:s}_{:s}".format(role, act))

        self.feature_names.extend([utils.BUYER+"_"+" ".join(x) for x in self.buyer_ngrams])
        self.feature_names.extend([utils.SELLER+"_"+" ".join(x) for x in self.seller_ngrams])

    @classmethod
    def add_first_agent_features(cls, ex, features):
        for e in ex['events']:
            if e['action'] == 'message':
                if ex['scenario']['kbs'][e['agent']]['personal']['Role'] == utils.BUYER:
                    features.append(0)
                else:
                    features.append(1)
                return

    @classmethod
    def add_category_features(cls, ex, categories, features):
        category = utils.get_category(ex)
        one_hot = [1 if k == category else 0 for k in categories]
        features.extend(one_hot)

    @classmethod
    def add_turn_features(cls, ex, features):
        turns = utils.get_turns_per_agent(ex)
        features.append(turns[0] + turns[1])

    @classmethod
    def add_token_features(cls, ex, roles, features):
        tokens = utils.get_total_tokens_per_agent(ex)
        features.append(tokens[roles[utils.BUYER]])
        features.append(tokens[roles[utils.SELLER]])

    @classmethod
    def add_price_features(cls, ex, roles, price_tracker, features):
        buyer_trend = StrategyAnalyzer.get_price_trend(price_tracker, ex, agent=roles[utils.BUYER])
        if len(buyer_trend) > 0:
            features.append(buyer_trend[0])
        else:
            features.append(-10.)  # indicate that buyer didn't mention any prices

        seller_trend = StrategyAnalyzer.get_price_trend(price_tracker, ex, agent=roles[utils.SELLER])
        if len(seller_trend) > 0:
            features.append(seller_trend[0])
        else:
            features.append(-10.)

    @classmethod
    def add_speech_act_features(cls, ex, price_tracker, features):
        buyer_acts = StrategyAnalyzer.get_speech_acts(ex, price_tracker, role=utils.BUYER)
        for act in SpeechActs.ACTS:
            frac = float(buyer_acts.count(act))/float(len(buyer_acts))
            features.append(frac)

        seller_acts = StrategyAnalyzer.get_speech_acts(ex, price_tracker, role=utils.SELLER)
        for act in SpeechActs.ACTS:
            frac = float(seller_acts.count(act))/float(len(seller_acts))
            features.append(frac)

    @classmethod
    def add_ngram_features(cls, ex, price_tracker, min_n, max_n, buyer_ngrams, seller_ngrams, features):
        for role in [utils.BUYER, utils.SELLER]:
            ngrams_list = buyer_ngrams if role == utils.BUYER else seller_ngrams
            found_ngrams = defaultdict(int)
            for i in xrange(min_n, max_n+1):
                extracted = cls.extract_ngrams(ex, role, price_tracker, i)
                for (ngram, count) in extracted.items():
                    found_ngrams[ngram] += count

            one_hot = [found_ngrams[k] if k in found_ngrams else 0 for k in ngrams_list]
            features.extend(one_hot)

    def create_feature_vector(self, ex):
        # order:
        # 0 or 1 (depending on whether first utterance is by buyer (0) or seller (1)),
        # category (one-hot vector)
        # # turns, # tokens (buyer), # tokens (seller), first price (buyer), first price (seller), speech
        # act features, buyer ngrams, seller ngrams

        features = []

        roles = {
            ex['scenario']['kbs'][0]['personal']['Role']: 0,
            ex['scenario']['kbs'][1]['personal']['Role']: 1
        }

        if self.first_agent_features:
            self.add_first_agent_features(ex, features)

        if self.category_features:
            self.add_category_features(ex, self.categories, features)

        if self.turn_features:
            self.add_turn_features(ex, features)

        if self.token_features:
            self.add_token_features(ex, roles, features)

        if self.price_features:
            self.add_price_features(ex, roles, self.price_tracker, features)

        if self.speech_act_features:
            self.add_speech_act_features(ex, self.price_tracker, features)

        if self.ngram_features:
            self.add_ngram_features(ex, self.price_tracker, self.min_n,
                                    self.max_n, self.buyer_ngrams, self.seller_ngrams, features)

        return features


    @classmethod
    def extract_ngrams(cls, ex, role, price_tracker, n):
        def _replace_prices(linked_tokens):
            new_tokens = []
            for token in linked_tokens:
                if isinstance(token, Entity) and token.canonical.type == 'price':
                    new_tokens.append(PRICE)
                else:
                    new_tokens.append(token)
            return new_tokens

        scenario = NegotiationScenario.from_dict(None, ex['scenario'])
        kbs = scenario.kbs
        roles = {
            0: kbs[0].facts['personal']['Role'],
            1: kbs[1].facts['personal']['Role']
        }

        ngrams = defaultdict(int)
        for event in ex['events']:
            if event['action'] != 'message' or roles[event['agent']] != role:
                continue
            raw_tokens = tokenize(event['data'])
            raw_tokens.append(SpecialSymbols.EOS)
            linked_tokens = price_tracker.link_entity(raw_tokens, kb=kbs[event['agent']])
            tokens = _replace_prices(linked_tokens)

            for i in xrange(0, len(tokens) - n + 1):
                ngram = tuple(tokens[i: i + n])
                ngrams[ngram] += 1

        return ngrams

    def extract_categories(self):
        for ex in self.transcripts:
            self.categories.add(utils.get_category(ex))

    def extract_ngram_features(self, role):
        ngrams_set = self.buyer_ngrams if role == utils.BUYER else self.seller_ngrams

        for ex in self.transcripts:
            for i in xrange(self.min_n, self.max_n+1):
                ngram_counts = self.extract_ngrams(ex, role, self.price_tracker, i)
                for (ngram, count) in ngram_counts.items():
                    ngrams_set[ngram] += count

    def write_feature_names(self, out_path):
        f = open(out_path, 'w')
        for name in self.feature_names:
            f.write("{:s}\n".format(name))
        f.close()


class MarginClassifier(object):
    POSITIVE_THRESHOLD = 0.7
    NEGATIVE_THRESHOLD = 0.3

    def __init__(self, transcripts, featurizer, split=0.8):
        self.featurizer = featurizer
        X = [featurizer.create_feature_vector(t) for t in transcripts]
        margins = [utils.get_margin(t, role=utils.BUYER) for t in transcripts]
        new_X = []
        y = []
        for (idx, margin) in enumerate(margins):
            if margin >= self.POSITIVE_THRESHOLD or margin < self.NEGATIVE_THRESHOLD:
                new_X.append(X[idx])
                y.append(margin)
        print "# of chats with buyer margin >= {:.1f}: {:d}".format(self.POSITIVE_THRESHOLD, y.count(1))
        print "# of chats with buyer margin < {:.1f}: {:d}".format(self.NEGATIVE_THRESHOLD, y.count(0))

        X, y = self.shuffle_data(X, y)
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None
        self.split(X, y, split=split)

        print "# of training examples: {:d}".format(self.train_X.shape[0])
        print "# of test examples: {:d}".format(self.test_X.shape[0])

        self.model = LogisticRegression(C=1.5)

    def shuffle_data(self, X, y):
        idxes = np.arange(len(X))
        np.random.shuffle(idxes)
        X = [X[i] for i in idxes]
        y = [y[i] for i in idxes]
        return X, y

    def split(self, X, y, split=0.8):
        num_train = int(len(X) * split)
        self.train_X = np.asarray(X[:num_train], dtype=np.float32)
        self.train_y = np.asarray(y[:num_train], dtype=np.float32)

        self.test_X = np.asarray(X[num_train:], dtype=np.float32)
        self.test_y = np.asarray(y[num_train:], dtype=np.float32)

    def train(self):
        self.model.fit(self.train_X, self.train_y)
        pred_y = self.model.predict(self.train_X)
        print "Train metrics:"
        self.print_metrics(self.train_y, pred_y)
        print "-----------------------"


    def test(self):
        pred_y = self.model.predict(self.test_X)
        print "Test metrics"
        self.print_metrics(self.test_y, pred_y)
        print "-----------------------"

    def print_metrics(self, true_y, pred_y):
        acc = np.mean(true_y == pred_y)
        print "Accuracy:\t{:.2f}".format(acc)

        tp = float(sum([1 for i in xrange(len(true_y)) if true_y[i] == 1 and pred_y[i] == 1]))
        fp = float(sum([1 for i in xrange(len(true_y)) if true_y[i] == 0 and pred_y[i] == 1]))
        fn = float(sum([1 for i in xrange(len(true_y)) if true_y[i] == 1 and pred_y[i] == 0]))

        print "Precision:\t{:.2f}".format(tp/(tp+fp))
        print "Recall:\t{:.2f}".format(tp/(tp+fn))

    def get_top_features(self, n=20):
        weights = self.model.coef_[0]
        print weights
        top_idxes = np.argsort(-np.fabs(weights))
        print "Top features:"
        for i in xrange(0, n):
            idx = top_idxes[i]
            name = self.featurizer.feature_names[idx]
            print "{:30s}\t{:.2f}".format(name, weights[idx])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output-dir', required=True, help='Directory containing all output from website')
    parser.add_argument('--limit', type=int, default=-1, help='Analyze the first N transcripts')
    parser.add_argument('--seed', default=1)
    add_price_tracker_arguments(parser)
    args = parser.parse_args()

    np.random.seed(args.seed)

    out_dir = args.output_dir
    if not os.path.exists(out_dir):
        raise ValueError("Output directory {:s} doesn't exist".format(out_dir))

    transcripts = json.load(open(os.path.join(out_dir, "transcripts", "transcripts.json"), 'r'))
    if args.limit > 0:
        transcripts = transcripts[:args.limit]

    featurizer = Featurizer(transcripts, args.price_tracker_model, max_n=3)
    predict_dir = os.path.join(out_dir, "predictions")
    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)

    featurizer.write_feature_names(os.path.join(predict_dir, "features.txt"))

    classifier = MarginClassifier(featurizer.transcripts, featurizer)
    classifier.train()
    classifier.test()
    classifier.get_top_features()

