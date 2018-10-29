import os
import re
import json
from collections import defaultdict
from itertools import izip, ifilter
from argparse import ArgumentParser
import numpy as np
from scipy import stats
import nltk.data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

from cocoa.core.util import read_json, write_json
from cocoa.core.entity import Entity, is_entity

from core.price_tracker import PriceTracker, PriceScaler, add_price_tracker_arguments
from core.tokenizer import tokenize
from core.scenario import Scenario
from analysis.html_visualizer import HTMLVisualizer
from dialogue import Dialogue
from liwc import LIWC
import utils

__author__ = 'anushabala'


THRESHOLD = 30.0
MAX_MARGIN = 2.4
MIN_MARGIN = -2.0
MIN_PRICE = -3
MAX_PRICE = 3


def round_partial(value, resolution=0.1):
    return round (value / resolution) * resolution

class StrategyAnalyzer(object):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def __init__(self, transcripts_paths, survey_paths, stats_path, price_tracker_model, liwc_path, max_examples=None):
        transcripts = self._read_transcripts(transcripts_paths, max_examples)
        self.dataset = utils.filter_rejected_chats(transcripts)

        dialogue_scores = self._read_surveys(survey_paths)

        self.price_tracker = PriceTracker(price_tracker_model)

        self.liwc = LIWC.from_pkl(liwc_path)

        # group chats depending on whether the seller or the buyer wins
        #self.buyer_wins, self.seller_wins = self.group_outcomes_and_roles()

        self.stats_path = stats_path
        if not os.path.exists(self.stats_path):
            os.makedirs(self.stats_path)

        self.examples = [Dialogue.from_dict(raw, dialogue_scores.get(raw['uuid'], {}), self.price_tracker) for raw in self.dataset]

    def _read_transcripts(self, transcripts_paths, max_examples):
        transcripts = []
        for transcripts_path in transcripts_paths:
            transcripts.extend(read_json(transcripts_path))
        if max_examples is not None:
            transcripts = transcripts[:max_examples]
        return transcripts

    def _read_surveys(self, survey_paths):
        dialogue_scores = {}
        for path in survey_paths:
            dialogue_scores.update(read_json(path)[1])
        return dialogue_scores

    def label_dialogues(self, labels=('speech_act', 'stage', 'liwc')):
        for dialogue in self.examples:
            if 'speech_act' in labels:
                dialogue.extract_keywords()
                dialogue.label_speech_acts()
            if 'stage' in labels:
                dialogue.label_stage()
            if 'liwc' in labels:
                dialogue.label_liwc(self.liwc)

    def summarize_tags(self):
        tags = defaultdict(lambda : defaultdict(int))
        for dialogue in self.examples:
            for turn in dialogue.turns:
                agent_name = dialogue.agents[turn.agent]
                for tag in turn.tags:
                    tags[agent_name][tag] += 1
        for system, labels in tags.iteritems():
            print system.upper()
            for k, v in labels.iteritems():
                print k, v

    def create_dataframe(self):
        data = []
        for dialogue in ifilter(lambda x: x.has_deal(), self.examples):
            for turn in dialogue.turns:
                for u in turn.iter_utterances():
                    row = {
                            'post_id': dialogue.post_id,
                            'chat_id': dialogue.chat_id,
                            'scenario_id': dialogue.scenario_id,
                            'buyer_target': dialogue.buyer_target,
                            'listing_price': dialogue.listing_price,
                            'margin_seller': dialogue.margins['seller'],
                            'margin_buyer': dialogue.margins['buyer'],
                            'stage': u.stage,
                            'role': turn.role,
                            'num_tokens': u.num_tokens(),
                            }
                    for a in u.speech_acts:
                        row['act_{}'.format(a[0].name)] = 1
                    for cat, word_count in u.categories.iteritems():
                        row['cat_{}'.format(cat)] = sum(word_count.values())
                    for q in dialogue.eval_questions:
                        for r in ('buyer', 'seller'):
                            key = 'eval_{question}_{role}'.format(question=q, role=r)
                            try:
                                row[key] = dialogue.eval_scores[r][q]
                            except KeyError:
                                row[key] = -1
                    data.append(row)
        df = pd.DataFrame(data).fillna(0)
        return df

    def summarize_liwc(self, k=10):
        categories = defaultdict(lambda : defaultdict(int))
        for dialogue in ifilter(lambda x: x.has_deal(), self.examples):
            for u in dialogue.iter_utterances():
                for cat, word_count in u.categories.iteritems():
                    for w, count in word_count.iteritems():
                        categories[cat][w] += count

        cat_freq = {c: sum(word_counts.values()) for c, word_counts in categories.iteritems()}
        cat_freq = sorted(cat_freq.items(), key=lambda x: x[1], reverse=True)
        def topk(word_counts, k=10):
            wc = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:k]
            return [x[0] for x in wc]
        for cat, count in cat_freq:
            print cat, count, topk(categories[cat], k)


    def html_visualize(self, output, img_path, css_file=None, mpld3_plugin=None):
        examples = [ex for ex in self.examples if self.has_deal(ex)]
        examples.sort(key=lambda d: (d.scenario_id, d.outcome['offer']['price']))

        include_scripts = []
        include_scripts.append('<script type="text/javascript" src="http://d3js.org/d3.v3.min.js"></script>')
        include_scripts.append('<script type="text/javascript" src="https://mpld3.github.io/js/mpld3.v0.2.js"></script>')
        if mpld3_plugin:
            with open(mpld3_plugin, 'r') as fin:
                mpld3_script = fin.read()
                include_scripts.append('<script type="text/javascript">{}</script>'.format(mpld3_script))

        css_style = """
            table {
                table-layout: fixed;
                width: 600px;
                border-collapse: collapse;
                }

            tr:nth-child(n) {
                border: solid thin;
                }

            .fig {
                height: 500px;
            }
            """
        if css_file:
            with open(css_file, 'r') as fin:
                css_style = '{}\n{}'.format(css_style, fin.read())
        style = ['<style type="text/css">', css_style, Dialogue.css, '</style>']

        header = ['<head>'] + style + include_scripts + ['</head>']

        plot_divs = []
        plot_scripts = []
        plot_scripts.append('<script type="text/javascript">')
        for d in examples:
            var_name = 'json_{}'.format(d.chat_id)
            json_str = json.dumps(d.fig_dict())
            div_name = 'fig_{}'.format(d.chat_id)
            plot_divs.append('<div class="fig" id="{div_name}"></div>'.format(div_name=div_name))
            plot_scripts.append('var {var_name} = {json_str};'.format(var_name=var_name, json_str=json_str))
            plot_scripts.append('!function(mpld3) {{ mpld3.draw_figure("{div_name}", {var_name}); }}(mpld3);'.format(div_name=div_name, var_name=var_name))
        plot_scripts.append('</script>')

        body = ['<body>']
        for d, plot_div in izip(examples, plot_divs):
            body.extend(
                NegotiationHTMLVisualizer.render_scenario(None, img_path=img_path, kbs=d.kbs, uuid=d.scenario_id)
                )
            body.append('<p>Final deal: {}</p>'.format(d.outcome['offer']['price']))
            body.append(plot_div)
        body.extend(plot_scripts)
        body.append('</body>')

        html_lines = ['<html>'] + header + body + ['</html>']

        outfile = open(output, 'w')
        for line in html_lines:
            outfile.write(line.encode('utf8')+"\n")
        print 'Write to', output
        outfile.close()

    @classmethod
    def get_price_trend(cls, price_tracker, chat, agent=None):
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
                linked_tokens = price_tracker.link_entity(raw_tokens,
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

    @classmethod
    def split_turn(cls, turn):
        # a single turn can be comprised of multiple sentences
        return cls.sent_tokenizer.tokenize(turn)

    def get_speech_acts(self, ex):
        stats = {0: [], 1: []}
        kbs = ex.kbs
        for e in ex.events:
            if e.action != 'message':
                continue

            sentences = self.split_turn(e.data.lower())

            for s in sentences:
                tokens = tokenize(s)
                linked_tokens = self.price_tracker.link_entity(tokens, kb=kbs[e.agent])
                act = SpeechActAnalyzer.get_speech_act(s, linked_tokens)
                stats[e.agent].append(act)

        return stats


    @classmethod
    def valid_price(cls, price):
        return price <= MAX_PRICE and price >= MIN_PRICE

    @classmethod
    def valid_margin(cls, margin):
        return margin <= MAX_MARGIN and margin >= MIN_MARGIN

    def get_first_price(self, ex):
        agents = {1: None, 0: None}
        for e in ex.events:
            if e.action == 'message':
                for sent_tokens in e.tokens:
                    for token in sent_tokens:
                        if agents[1] and agents[0]:
                            return agents
                        # Return at the first mention
                        if is_entity(token):
                            price = token.canonical.value
                            agents[e.agent] = (e.role, price)
                            return agents
        return agents

    @classmethod
    def get_margin(cls, ex, price, agent, role, remove_outlier=True):
        agent_target = ex.scenario.kbs[agent].facts["personal"]["Target"]
        partner_target = ex.scenario.kbs[1 - agent].facts["personal"]["Target"]
        midpoint = (agent_target + partner_target) / 2.
        norm_factor = np.abs(midpoint - agent_target)
        if role == utils.SELLER:
            margin = (price - midpoint) / norm_factor
        else:
            margin = (midpoint - price) / norm_factor
        if remove_outlier and not cls.valid_margin(margin):
            return None
        return margin

    @classmethod
    def print_ex(cls, ex):
        print '===================='
        for e in ex.events:
            print e.role.upper(), e.data
        print '===================='

    def get_basic_stats(self, ex):
        stats = {0: None, 1: None}
        for agent in (0, 1):
            num_turns = ex.num_turns()
            num_tokens = ex.num_tokens()
            stats[agent] = {
                    'role': ex.kbs[agent].facts['personal']['Role'],
                    'num_turns': num_turns,
                    'num_tokens_per_turn': num_tokens / num_turns * 1.,
                    }
        return stats

    def is_good_negotiator(self, final_margin):
        if final_margin > 0.8 and final_margin <= 1:
            return 1
        elif final_margin < -0.8 and final_margin <= -1:
            return -1
        else:
            return 0

    @classmethod
    def has_deal(cls, ex):
        if ex.outcome is None or ex.outcome['reward'] == 0 or ex.outcome.get('offer', None) is None or ex.outcome['offer']['price'] is None:
            return False
        return True

    def plot_speech_acts(self, output='figures/speech_acts'):
        data = defaultdict(list)
        for ex in ifilter(self.has_deal, self.examples):
            stats = self.get_speech_acts(ex)
            final_price = ex.outcome['offer']['price']
            for agent, acts in stats.iteritems():
                role = ex.agent_to_role[agent]
                final_margin = self.get_margin(ex, final_price, agent, role)
                label = self.is_good_negotiator(final_margin)
                for act in acts:
                    data['role'].append(role)
                    data['label'].append(label)
                    data['final_margin'].append(final_margin)
                    data['act'].append(act)

        for role in ('seller', 'buyer'):
            print role.upper()
            print '='*40
            good_seller_act = [a for r, l, m, a in izip(data['role'], data['label'], data['final_margin'], data['act']) if r == role and l == 1]
            bad_seller_act = [a for r, l, m, a in izip(data['role'], data['label'], data['final_margin'], data['act']) if r == role and l == -1]
            sum_act = lambda a, l: np.mean([1 if a in x else 0 for x in l])
            print len(good_seller_act), len(bad_seller_act)
            print '{:<20} {:<10} {:<10}'.format('ACT', 'GOOD', 'BAD')
            print '-'*40
            for act in SpeechActs.ACTS:
                print '{:<20} {:<10.4f} {:<10.4f}'.format(act, sum_act(act, good_seller_act), sum_act(act, bad_seller_act))

        return


    def plot_basic_stats(self, output='figures/basic_stats'):
        data = {'role': [], 'final_margin': [], 'num_turns': [], 'num_tokens_per_turn': [], 'label': []}
        for ex in ifilter(self.has_deal, self.examples):
            stats = self.get_basic_stats(ex)
            final_price = ex.outcome['offer']['price']
            for agent, stats in stats.iteritems():
                role = stats['role']
                final_margin = self.get_margin(ex, final_price, agent, role)
                label = self.is_good_negotiator(final_margin)
                for k, v in stats.iteritems():
                    data[k].append(v)
                data['label'].append(label)
                data['final_margin'].append(final_margin)
        fig = plt.figure()
        df = pd.DataFrame(data)
        #g = sns.lmplot(x='num_tokens_per_turn', y='final_margin', col='role', row='label', data=dataframe, scatter_kws={'alpha':0.5})
        #g.savefig(output)
        for role in ('buyer', 'seller'):
            d1 = df.num_tokens_per_turn[(df['label'] == 1) & (df['role'] == role)]
            d2 = df.num_tokens_per_turn[(df.label == -1) & (df.role == role)]
            sns.distplot(d1, label='good')
            sns.distplot(d2, label='bad')
            plt.legend()
            plt.savefig('%s_%s.png' % (output, role))
            plt.clf()

    def plot_opening_vs_result(self, output='figures/opening_vs_result.png'):
        data = {'role': [], 'init_margin': [], 'final_margin': []}
        for ex in ifilter(self.has_deal, self.examples):
            final_price = ex.outcome['offer']['price']
            init_prices = self.get_first_price(ex)
            for agent, p in init_prices.iteritems():
                if p is None:
                    continue
                role, price = p
                init_margin = self.get_margin(ex, price, agent, role)
                final_margin = self.get_margin(ex, final_price, agent, role)
                if init_margin is None or final_margin is None:
                    continue
                # NOTE: sometimes one is saying a price is not okay, i.e. negative mention
                # TODO: detect negative vs positive mention
                if init_margin == -1 and init_margin < final_margin:
                    continue
                #if init_margin < final_margin:
                #    print role, (price, init_margin), (final_price, final_margin)
                #    self.print_ex(ex)
                #    import sys; sys.exit()
                for k, v in izip(('role', 'init_margin', 'final_margin'), (role, init_margin, final_margin)):
                    data[k].append(v)
        dataframe = pd.DataFrame(data)
        fig = plt.figure()
        g = sns.lmplot(x='init_margin', y='final_margin', col='role', data=dataframe, scatter_kws={'alpha':0.5})
        g.savefig(output)

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
                margin = utils.get_margin(ex)
                if margin > MAX_MARGIN or margin < 0.:
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

    def plot_margin_histograms(self):
        for (lbl, group) in zip(['buyer_wins', 'seller_wins'], [self.buyer_wins, self.seller_wins]):
            margins = []
            for ex in group:
                winner = utils.get_winner(ex)
                if winner is None:
                    continue
                margin = utils.get_margin(ex)
                if 0 <= margin <= MAX_MARGIN:
                    margins.append(margin)

            b = np.linspace(0, MAX_MARGIN, num=int(MAX_MARGIN/0.2)+2)
            print b
            hist, bins = np.histogram(margins, bins=b)

            width = np.diff(bins)
            center = (bins[:-1] + bins[1:]) / 2

            fig, ax = plt.subplots(figsize=(8,3))
            ax.bar(center, hist, align='center', width=width)
            ax.set_xticks(bins)

            save_path = os.path.join(self.stats_path, '{:s}_wins_margins_histogram.png'.format(lbl))
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

    def plot_price_trends(self, top_n=10):
        labels = ['buyer_wins', 'seller_wins']
        for (group, lbl) in zip([self.buyer_wins, self.seller_wins], labels):
            plt.figure(figsize=(10, 6))
            trends = []
            for chat in group:
                winner = utils.get_winner(chat)
                margin = utils.get_margin(chat)
                if margin > 1.0 or margin < 0.:
                    continue
                if winner is None:
                    continue

                # print "Winner: Agent {:d}\tWin margin: {:.2f}".format(winner, margin)
                if winner == -1 or winner == 0:
                    trend = self.get_price_trend(self.price_tracker, chat, agent=0)
                    if len(trend) > 1:
                        trends.append((margin, chat, trend))
                if winner == -1 or winner == 1:
                    trend = self.get_price_trend(self.price_tracker, chat, agent=1)
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
                    if isinstance(token, Entity) and token.canonical.type == 'price':
                        prices += 1

        return prices

    def plot_speech_acts_old(self):
        labels = ['buyer_wins', 'seller_wins']
        for (group, lbl) in zip([self.buyer_wins, self.seller_wins], labels):
            plt.figure(figsize=(10, 6))
            speech_act_counts = dict((act, defaultdict(list)) for act in SpeechActs.ACTS)
            for chat in group:
                winner = utils.get_winner(chat)
                margin = utils.get_margin(chat)
                if margin > MAX_MARGIN or margin < 0.:
                    continue
                if winner is None:
                    continue

                margin = round_partial(margin) # round the margin to the nearest 0.1 to reduce noise

                if winner == -1 or winner == 0:
                    speech_acts = self.get_speech_acts(chat, agent=0)
                    # print "Chat {:s}\tWinner: {:d}".format(chat['uuid'], winner)
                    # print speech_acts
                    for act in SpeechActs.ACTS:
                        frac = float(speech_acts.count(act))/float(len(speech_acts))
                        speech_act_counts[act][margin].append(frac)
                if winner == -1 or winner == 1:
                    speech_acts = self.get_speech_acts(chat, agent=1)
                    # print "Chat {:s}\tWinner: {:d}".format(chat['uuid'], winner)
                    # print speech_acts
                    for act in SpeechActs.ACTS:
                        frac = float(speech_acts.count(act))/float(len(speech_acts))
                        speech_act_counts[act][margin].append(frac)

            for act in SpeechActs.ACTS:
                counts = speech_act_counts[act]
                margins = []
                fracs = []
                errors = []
                bin_totals = 0.
                for m in sorted(counts.keys()):
                    if len(counts[m]) > THRESHOLD:
                        bin_totals += len(counts[m])
                        margins.append(m)
                        fracs.append(np.mean(counts[m]))
                        errors.append(stats.sem(counts[m]))
                print bin_totals / float(len(margins))

                plt.errorbar(margins, fracs, yerr=errors, label=act, fmt='--o')

            plt.xlabel('Margin of victory')
            plt.ylabel('Fraction of speech act occurences')
            plt.title('Speech act frequency vs. margin of victory')
            plt.legend()
            save_path = os.path.join(self.stats_path, '{:s}_speech_acts.png'.format(lbl))
            plt.savefig(save_path)

    def plot_speech_acts_by_role(self):
        labels = utils.ROLES
        for lbl in labels:
            plt.figure(figsize=(10, 6))
            speech_act_counts = dict((act, defaultdict(list)) for act in SpeechActs.ACTS)
            for chat in self.dataset:
                if utils.get_winner(chat) is None:
                    # skip chats with no outcomes
                    continue
                speech_acts = self.get_speech_acts(chat, role=lbl)
                agent = 1 if chat['scenario']['kbs'][1]['personal']['Role'] == lbl else 0
                margin = utils.get_margin(chat, agent=agent)
                if margin > MAX_MARGIN:
                    continue
                margin = round_partial(margin)
                for act in SpeechActs.ACTS:
                    frac = float(speech_acts.count(act))/float(len(speech_acts))
                    speech_act_counts[act][margin].append(frac)

            for act in SpeechActs.ACTS:
                counts = speech_act_counts[act]
                margins = []
                fracs = []
                errors = []
                for m in sorted(counts.keys()):
                    if len(counts[m]) > THRESHOLD:
                        margins.append(m)
                        fracs.append(np.mean(counts[m]))
                        errors.append(stats.sem(counts[m]))

                plt.errorbar(margins, fracs, yerr=errors, label=act, fmt='--o')

            plt.xlabel('Margin of victory')
            plt.ylabel('Fraction of speech act occurences')
            plt.title('Speech act frequency vs. margin of victory')
            plt.legend()
            save_path = os.path.join(self.stats_path, '{:s}_speech_acts.png'.format(lbl))
            plt.savefig(save_path)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output-dir', required=True, help='Directory containing all output from website')
    parser.add_argument('--max-examples', type=int, default=100, help='Maximum number of examples to run')
    parser.add_argument('--html-visualize', action='store_true', help='Output html files')
    parser.add_argument('--mpld3-plugin', default=None, help='Javascript of the mpld3 plugin')
    add_price_tracker_arguments(parser)
    HTMLVisualizer.add_html_visualizer_arguments(parser)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        raise ValueError("Output directory {:s} doesn't exist".format(args.output_dir))

    transcripts_path = [os.path.join(args.output_dir, 'transcripts', 'transcripts.json')]
    surveys_path = [os.path.join(args.output_dir, 'transcripts', 'surveys.json')]
    stats_output = os.path.join(args.output_dir, 'stats')

    analyzer = StrategyAnalyzer(transcripts_path, surveys_path, stats_output, args.price_tracker_model, 'data/liwc.pkl', args.max_examples)
    analyzer.create_dataframe()

    if args.html_visualize:
        analyzer.label_dialogues(('speech_act',))
        analyzer.html_visualize(args.html_output, args.img_path, args.css_file, args.mpld3_plugin)

    #analyzer.plot_opening_vs_result()
    #analyzer.plot_basic_stats()
    #analyzer.plot_speech_acts()

    # analyzer.plot_length_histograms()
    # analyzer.plot_margin_histograms()
    # analyzer.plot_length_vs_margin()
    # analyzer.plot_price_trends()
    #analyzer.plot_speech_acts()
    #analyzer.plot_speech_acts_by_role()
