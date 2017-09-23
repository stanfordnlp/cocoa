from pygtrie import Trie
from nltk.corpus import stopwords
import numpy as np

from cocoa.systems.rulebased_system import RulebasedSystem as BaseRulebasedSystem
from cocoa.core.dataset import read_examples
from cocoa.core.entity import is_entity

from core.scenario import Scenario
from core.tokenizer import detokenize
from analysis.dialogue import Utterance, Dialogue
from analysis.speech_acts import SpeechActAnalyzer
from model.preprocess import Preprocessor
from sessions.rulebased_session import RulebasedSession

def add_rulebased_arguments(parser):
    parser.add_argument('--templates', help='Path to templates (.pkl)')

class RulebasedSystem(BaseRulebasedSystem):

    def __init__(self, lexicon, timed_session=False, templates=None):
        super(RulebasedSystem, self).__init__(timed_session)
        self.lexicon = lexicon
        self.templates = templates

    def _new_session(self, agent, kb, config=None):
        return RulebasedSession.get_session(agent, kb, self.lexicon, config, self.templates)

class TemplateExtractor(object):
    stopwords = set(stopwords.words('english'))
    stopwords.update(['may', 'might', 'rent', 'new', 'brand', 'low', 'high', 'now', 'available'])

    def __init__(self, price_tracker):
        self.price_tracker = price_tracker

    def tag_utterance(self, utterance):
        """Get speech acts of the utterance.
        """
        acts = []
        if SpeechActAnalyzer.is_question(utterance):
            acts.append('question')
        if utterance.prices:
            acts.append('price')
        if SpeechActAnalyzer.is_price(utterance):
            acts.append('vague-price')
        return acts

    def tag_utterances(self, utterances, listing_price):
        """Tag utterances in the dialogue context.

        Args:
            utterances ([Utterance])
            listing_price (float)

        """
        tagged_utterances = []
        N = len(utterances)
        prev_acts = []
        mentioned_price = False
        for i, utterance in enumerate(utterances):
            if utterance.action != 'message':
                continue
            acts = self.tag_utterance(utterance)
            tag = None
            if i == 0 and not 'price' in acts:
                tag = 'intro'
            elif 'price' in acts and not mentioned_price:
                mentioned_price = True
                tag = 'init-price'
            elif i + 1 < N and utterances[i+1].action == 'offer' and not acts:
                tag = 'agree'
            elif (not 'price' in acts) and 'vague-price' in acts:
                tag = 'vague-price'
            elif 'price' in acts and 'price' in prev_acts:
                prev_utterance = utterances[i-1]
                # TODO: more rules for price labeling/parsing
                if len(utterance.prices) == 1 and len(prev_utterance.prices) == 1:
                    if utterance.prices[0].canonical.value != prev_utterance.prices[0].canonical.value and utterance.prices[0].canonical.value != listing_price:
                        tag = 'counter-price'
                    else:
                        tag = 'agree-price'
            elif acts == ['question']:
                tag = 'inquiry'
            elif not acts and prev_acts == ['question']:
                tag = 'inform'
            if tag:
                tagged_utterances.append((i, tag, utterance))

            prev_acts = acts

        return tagged_utterances

    def make_template(self, utterance, kb):
        """Abstract product and price.
        """
        title = kb.facts['item']['Title'].lower().split()
        new_tokens = []
        num_title = 0
        num_price = 0
        for token in utterance.tokens:
            if token in title and not token in self.stopwords:
                if len(new_tokens) > 0 and new_tokens[-1] == '{title}':
                    continue
                else:
                    num_title += 1
                    new_tokens.append('{title}')
            elif is_entity(token):
                num_price += 1
                new_tokens.append('${price}')
            else:
                new_tokens.append(token)

        if num_price > 1 or num_title > 1:
            return None
        template = detokenize(new_tokens)
        return template

    def extract_templates(self, transcripts_paths, max_examples=-1):
        templates = Trie()
        examples = read_examples(transcripts_paths, max_examples, Scenario)
        for example in examples:
            if Preprocessor.skip_example(example):
                continue
            kbs = example.scenario.kbs
            events = example.events
            category = kbs[0].category
            roles = {agent_id: kbs[agent_id].role for agent_id in (0, 1)}

            utterances = []
            for event in events:
                if event.action == 'message':
                    utterances.append(Utterance.from_text(event.data, self.price_tracker, kbs[event.agent]))
                elif event.action == 'offer':
                    if Dialogue.has_deal(example.outcome) and Dialogue.agreed_deal(example.outcome):
                        try:
                            utterances.append(Utterance(str(event.data['price']), [], event.action))
                        # Not sure why - sometimes offer event is None and is not the final outcome
                        except ValueError:
                            pass
            tagged_utterances = self.tag_utterances(utterances, listing_price=kbs[0].listing_price)

            for i, tag, utterance in tagged_utterances:
                agent = events[i].agent
                template = self.make_template(utterance, kbs[agent])
                role = roles[agent]
                if template:
                    key = (category, role, tag)
                    if key in templates:
                        templates[key].append(template)
                    else:
                        templates[key] = [template]
        templates = self.add_counts(templates, n=4)
        return templates

    def ngrams(self, tokens, n=1):
        for i in xrange(max(1, len(tokens)-n+1)):
            yield tokens[i:i+n]

    def add_counts(self, templates, n=4):
        ngram_counter = Trie()
        for k, temps in templates.iteritems():
            for temp in temps:
                tokens = temp.split()
                for ngram in self.ngrams(tokens, n):
                    if ngram not in ngram_counter:
                        ngram_counter[ngram] = 1
                    else:
                        ngram_counter[ngram] += 1
        for k, temps in templates.iteritems():
            for i, temp in enumerate(temps):
                tokens = temp.split()
                counts = []
                for ngram in self.ngrams(tokens, n):
                    counts.append(ngram_counter[ngram])
                if not counts:
                    print tokens
                    import sys; sys.exit()
                mean_count = np.mean(counts)
                temps[i] = (mean_count, temp)
        return templates

############# TEST #############
if __name__ == '__main__':
    import argparse
    from cocoa.core.util import write_pickle
    from core.price_tracker import PriceTracker

    parser = argparse.ArgumentParser()
    parser.add_argument('--transcripts', nargs='*')
    parser.add_argument('--price-tracker-model')
    parser.add_argument('--max-examples', default=-1, type=int)
    parser.add_argument('--output')
    args = parser.parse_args()

    price_tracker = PriceTracker(args.price_tracker_model)
    template_extractor = TemplateExtractor(price_tracker)
    templates = template_extractor.extract_templates(args.transcripts, args.max_examples)
    write_pickle(templates, args.output)

    for k, v in templates.iteritems():
        print k
        for u in v[:10]:
            print u
