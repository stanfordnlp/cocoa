import math
import json
import re
from collections import defaultdict
from itertools import izip, ifilter
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cocoa.core.dataset import Example
from cocoa.core.entity import Entity, is_entity, CanonicalEntity

from core.scenario import Scenario
from core.tokenizer import tokenize
from speech_acts import SpeechActAnalyzer

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

class DialogueStage(object):
    INQUIRE = 0
    DISCUSS = 1
    CONCLUDE = 2

class Utterance(object):
    def __init__(self, raw_text, tokens, action='message'):
        self.text = raw_text
        self.tokens = tokens
        self.action = action
        self.prices = []
        self.keywords = []
        self.speech_acts = []
        self.stage = -1
        self.categories = defaultdict(lambda : defaultdict(int))

        if self.action == 'message':
            self.prices = [token for token in self.tokens if is_entity(token)]
        elif self.action == 'offer':
            price = self.text
            self.prices.append(Entity(price, CanonicalEntity(float(price), 'price')))

    @classmethod
    def from_text(cls, raw_text, price_tracker, kb):
        tokens = price_tracker.link_entity(tokenize(raw_text), kb=kb, scale=False)
        return cls(raw_text, tokens)

    def num_tokens(self):
        return len(self.tokens) if self.action == 'message' else 0

    def extract_keywords(self):
        if self.action == 'message':
            # Re-tokenize here because for tagging is case-sensitive
            tags = pos_tag(tokenize(self.text, lowercase=False))
            self.keywords = [word for word, tag in tags if re.match(r'NN*|ADJ*', tag)]

    def __str__(self):
        if self.action == 'message':
            tokens = 'tokens: {}'.format(' '.join([str(x) for x in self.tokens]))
            keywords = 'keywords: {}'.format(' | '.join(self.keywords))
            return '\n'.join([tokens, keywords])
        elif self.action == 'offer':
            return '<offer> {}'.format(self.text)
        else:
            return '<{}>'.format(self.action)

class Turn(object):
    def __init__(self, agent, role, action, tags, utterances):
        self.agent = agent
        self.role = role
        self.action = action
        self.tags = tags
        self.utterances = utterances

    def __str__(self):
        lines = []
        lines.append(self.role.upper())
        for u in self.utterances:
            lines.append(str(u))
        return '\n'.join(lines)

    @classmethod
    def from_event(cls, event, kbs, price_tracker):
        kb = kbs[event.agent]
        if event.action == 'message':
            sents = sent_tokenizer.tokenize(event.data)
            utterances = [Utterance.from_text(sent, price_tracker, kb) for sent in sents]
        elif event.action == 'offer':
            utterances = [Utterance(str(event.data['price']), [], event.action)]
        else:
            utterances = [Utterance(event.action, [], event.action)]
        role = kb.facts['personal']['Role']
        return cls(event.agent, role, event.action, event.tags, utterances)

    def num_tokens(self):
        return sum([u.num_tokens() for u in self.utterances])

    def extract_keywords(self):
        for u in self.utterances:
            u.extract_keywords()

    def iter_prices(self):
        for u in self.utterances:
            for p in u.prices:
                yield p

    def iter_keywords(self):
        for u in self.utterances:
            for w in u.keywords:
                yield w

    def iter_utterances(self):
        for u in self.utterances:
            yield u

class Dialogue(object):
    css = """
        .utterance {
            width: 200px;
            border: 2px solid green;
            padding: 2px;
            background: #FFFFFF;
        }
    """

    scenario_map = {}

    eval_questions = ('fluent', 'coherent', 'persuasive', 'fair')

    def __init__(self, chat_id, scenario_id, post_id, kbs, turns, outcome, scores, agents):
        self.chat_id = chat_id
        self.post_id = post_id
        self.kbs = kbs
        self.agents = agents
        self.kb_by_role = {kb.facts['personal']['Role']: kb for kb in kbs}
        self.listing_price = self.kb_by_role['seller'].facts['personal']['Target']
        self.buyer_target = self.kb_by_role['buyer'].facts['personal']['Target']
        self.scenario_id = self.get_scenario_id(post_id, self.buyer_target)
        self.turns = turns
        self.outcome = outcome
        self.eval_scores = {kbs[agent_id].facts['personal']['Role']: s for agent_id, s in scores.iteritems()}
        self.margins = self.compute_margin()

    @classmethod
    def get_scenario_id(cls, post_id, buyer_target):
        key = (post_id, buyer_target)
        if not key in cls.scenario_map:
            val = len(cls.scenario_map)
            cls.scenario_map[key] = val
        else:
            val = cls.scenario_map[key]
        return val

    @classmethod
    def agreed_deal(cls, outcome):
        if outcome and outcome.get('reward') == 1:
            return True
        return False

    @classmethod
    def has_deal(cls, outcome):
        if outcome is None or outcome.get('offer') is None or outcome['offer'].get('price') is None:
            return False
        return True

    def compute_margin(self):
        margins = {'seller': None, 'buyer': None}
        if not self.has_deal(self.outcome):
            return margins

        targets = {}
        for role in ('seller', 'buyer'):
            targets[role] = self.kb_by_role[role].facts["personal"]["Target"]
        midpoint = (targets['seller'] + targets['buyer']) / 2.
        price = self.outcome['offer']['price']

        norm_factor = abs(midpoint - targets['seller'])
        margins['seller'] = (price - midpoint) / norm_factor
        # Zero sum
        margins['buyer'] = -1. * margins['seller']
        return margins

    def __str__(self):
        self.kb_by_role['seller'].dump()
        return '\n'.join([str(t) for t in self.turns])

    @classmethod
    def is_valid_event(cls, event):
        if event.action == 'message':
            if len(event.data.strip()) == 0:
                return False
        if event.action == 'offer':
            if math.isnan(event.data['price']):
                return False
        return True

    @classmethod
    def from_dict(cls, raw_chat, raw_scores, price_tracker):
        ex = Example.from_dict(None, raw_chat, Scenario)
        kbs = ex.scenario.kbs
        turns = [Turn.from_event(event, kbs, price_tracker) for event in ex.events if cls.is_valid_event(event)]
        scores = cls.parse_scores(raw_scores)
        return cls(ex.ex_id, ex.uuid, ex.scenario.post_id, kbs, turns, ex.outcome, scores, ex.agents)

    @classmethod
    def parse_scores(cls, raw_scores):
        agent_scores = {}
        for agent_id, scores in raw_scores.iteritems():
            agent_id = int(agent_id)
            question_scores = {}
            for question, score in scores.iteritems():
                if question in cls.eval_questions:
                    question_scores[question] = int(score)
            agent_scores[agent_id] = question_scores
        return agent_scores

    def num_tokens(self):
        return sum([t.num_tokens() for t in self.turns])

    def num_turns(self):
        return len(self.turns)

    def iter_utterances(self):
        for turn in self.turns:
            for u in turn.iter_utterances():
                yield u

    def extract_keywords(self):
        for turn in self.turns:
            turn.extract_keywords()

    def label_speech_acts(self):
        prev_turn = None
        for turn in self.turns:
            for utterance in turn.utterances:
                utterance.speech_acts = SpeechActAnalyzer.get_speech_acts(utterance, prev_turn)
            prev_turn = turn

    def label_stage(self):
        listing_price = self.kbs[0].facts['item']['Price']
        try:
            final_price = self.outcome['offer']['price']
        except (TypeError, KeyError) as e:
            final_price = None
        prev_stage = DialogueStage.INQUIRE
        for u in self.iter_utterances():
            stage = None
            for price in u.prices:
                if prev_stage == DialogueStage.INQUIRE and price.canonical.value != listing_price:
                    stage = DialogueStage.DISCUSS
                    break
                elif price.canonical.value == final_price:
                    stage = DialogueStage.CONCLUDE
                    break
            if not stage:
                stage = prev_stage
            u.stage = stage
            prev_stage = stage

    def _treebank_to_liwc_token(self, tokens):
        '''
        In LIWC dictinoary, "'re", "n't" etc are not separated.
        '''
        new_tokens = []
        for token in tokens:
            if not is_entity(token) and (token.startswith("'") or token == "n't") and len(new_tokens) > 0 and not is_entity(new_tokens[-1]):
                new_tokens[-1] += token
            else:
                new_tokens.append(token)
        return new_tokens

    def label_liwc(self, liwc):
        for utterance in self.iter_utterances():
            if utterance.action == 'message':
                tokens = self._treebank_to_liwc_token(utterance.tokens)
                for token in ifilter(lambda x: not is_entity(x), tokens):
                    cats = liwc.lookup(token)
                    for cat in cats:
                        utterance.categories[cat][token] += 1

    def fig_dict(self):
        '''
        Plot price trend and utterances by mpld3 and return the json dict for html rendering.
        '''
        import mpld3
        data = {k: [] for k in ('time_step', 'price', 'speech_acts', 'text', 'role')}
        turn_boundary = []
        stage_boundary = []
        t = 0
        curr_price = self.kbs[0].facts['item']['Price']
        prev_stage = None
        for turn in self.turns:
            for utterance in turn.utterances:
                t += 1
                if utterance.prices:
                    curr_price = utterance.prices[-1].canonical.value
                data['time_step'].append(t)
                data['price'].append(curr_price)
                data['speech_acts'].append(utterance.speech_acts)
                data['text'].append(utterance.text)
                data['role'].append(turn.role)
                if utterance.stage != prev_stage and t > 1:
                    stage_boundary.append(t)
                prev_stage = utterance.stage
            turn_boundary.append(t)

        fig, ax = plt.subplots()
        fig.set_size_inches(15, 5)

        seller_target = self.kb_by_role['seller'].facts['personal']['Target']
        buyer_target = self.kb_by_role['buyer'].facts['personal']['Target']
        v_dist = seller_target - buyer_target
        v_offset = max(0.5, v_dist / 10)
        min_price = min(data['price'])
        max_price = max(data['price'])
        y_min = min(min_price, buyer_target) - 4*v_offset
        y_max = max(max_price, seller_target) + 4*v_offset
        ax.set_ylim(y_min, y_max)

        # Plot line
        ax.plot(data['time_step'], data['price'], zorder=10)

        # Label points
        offset = (y_max - y_min) / 20
        for i, (t, p, a) in enumerate(izip(data['time_step'], data['price'], data['speech_acts'])):
            v = (1 if i % 2 == 0 else -1) * offset
            ax.text(t, p+v, '|'.join([x[0].abrv for x in a]), fontsize=15, horizontalalignment='center', verticalalignment='center')

        # Draw turn boundary
        def hline(ax, pos, min_, max_, style, **kwargs):
            ax.plot([min_, max_], [pos, pos], style, **kwargs)
        def vline(ax, pos, min_, max_, style, **kwargs):
            ax.plot([pos, pos], [min_, max_], style, **kwargs)
        for b in turn_boundary:
            vline(ax, (b+0.5), y_min, y_max, 'b--', alpha=0.5, zorder=0)
        # Draw stage boundary
        for b in stage_boundary:
            vline(ax, (b-0.5), y_min, y_max, 'k-')
        # Draw target prices
        N = len(data['time_step'])
        for kb in self.kbs:
            target = kb.facts['personal']['Target']
            hline(ax, target, 0, N+1, 'r--')

        # Scatter seller points
        for role, color in izip(('seller', 'buyer'), ('r', 'b')):
            time_step = [x for x, r in izip(data['time_step'], data['role']) if r == role]
            price = [x for x, r in izip(data['price'], data['role']) if r == role]
            points = ax.scatter(time_step, price, s=200, zorder=20)
            labels = ['<div class="utterance">{}</div>'.format(u) for u, r in izip(data['text'], data['role']) if r == role]
            tooltip = mpld3.plugins.PointHTMLTooltip(points, labels=labels, voffset=10, hoffset=10, css=self.css)
            mpld3.plugins.connect(fig, tooltip)

        fig_dict = mpld3.fig_to_dict(fig)
        plt.close()
        return fig_dict

####### TEST #########
if __name__ == '__main__':
    from cocoa.core.negotiation.price_tracker import PriceTracker
    from cocoa.core.util import read_json
    from liwc import LIWC

    transcripts = read_json('web_output/combined/transcripts/transcripts.json')
    price_tracker = PriceTracker('/scr/hehe/game-dialogue/price_tracker.pkl')
    liwc = LIWC.from_pkl('data/liwc.pkl')
    dialogue = Dialogue.from_dict(transcripts[0], price_tracker)
    dialogue.label_liwc(liwc)
    for u in dialogue.iter_utterances():
        print u.text
        print u.categories
    #dialogue.extract_keywords()
    #dialogue.label_speech_acts()
    #dialogue.label_stage()
    #dialogue.fig_dict()
