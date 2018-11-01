from nltk.corpus import stopwords
from pygtrie import Trie
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from cocoa.core.dataset import read_examples
from cocoa.core.entity import is_entity
from cocoa.core.util import read_pickle, write_json

from core.scenario import Scenario
from core.tokenizer import detokenize
from analysis.dialogue import Utterance, Dialogue
from analysis.speech_acts import SpeechActAnalyzer
from neural.preprocess import Preprocessor

class Templates(object):
    def __init__(self, templates):
        self.templates = pd.DataFrame(templates)
        self.vectorizer = TfidfVectorizer()
        self.build_tfidf()

    @classmethod
    def from_pickle(cls, path):
        templates = read_pickle(path)
        return cls(templates)

    def build_tfidf(self):
        # TODO: context + response?
        documents = self.templates['context'].values
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

    def search(self, context, category=None, role=None, context_tag=None, response_tag=None, used_templates=None, T=1.):
        loc = self.get_filter(category=category, role=role, context_tag=context_tag, response_tag=response_tag, used_templates=used_templates).values
        features = self.vectorizer.transform([context])
        scores = self.tfidf_matrix * features.T

        scores = scores.todense()[loc]
        rows = self.templates[loc]

        scores = np.squeeze(np.array(scores))
        ids = np.argsort(scores)[::-1][:20]
        rows = rows.iloc[ids]
        counts = rows['count'].values
        return self.sample(counts, rows, T=T)

    def softmax(self, scores, T=1.):
        exp_scores = np.exp((scores - np.max(scores)) / T)
        return exp_scores / np.sum(exp_scores)

    def sample(self, counts, templates, T=1.):
        probs = self.softmax(counts, T=T)
        template_id = np.random.multinomial(1, probs).argmax()
        template = templates.iloc[template_id]
        return template

    def get_filter(self, category=None, role=None, context_tag=None, response_tag=None, used_templates=None):
        locs = []
        loc = True
        if used_templates:
            loc = (~self.templates.id.isin(used_templates))
            locs.append(loc)
        assert category and role
        loc = loc & (self.templates.category == category) & (self.templates.role == role)
        locs.append(loc)
        if response_tag:
            loc = loc & (self.templates.response_tag == response_tag)
            locs.append(loc)
        if context_tag:
            loc = loc & (self.templates.context_tag == context_tag)
            locs.append(loc)
        for loc in locs[::-1]:
            if np.sum(loc) > 0:
                return loc
        return locs[0]

    def choose(self, used_templates=None, category=None, role=None, context_tag=None, response_tag=None, T=1.):
        loc = self.get_filter(category=category, role=role, context_tag=context_tag, response_tag=response_tag, used_templates=used_templates)
        templates = self.templates[loc]
        if len(templates) > 0:
            counts = templates['count'].values
            return self.sample(counts, templates, T)

        print 'WARNING: no available templates found, returning a random one'
        counts = self.templates['count'].values
        return self.sample(counts, self.templates, T)

        #for loc in locs[::-1]:
        #    templates = self.templates[loc][['id', 'count', 'response']].values
        #    if len(templates) > 0:
        #        return self.sample(templates, T)
        #print 'WARNING: no available templates found, returning a random one'
        #return self.sample(self.templates[['id', 'count', 'response']].values, T)

    def dump(self, n=-1):
        df = self.templates.groupby(['category', 'role', 'context_tag', 'response_tag'])
        for group in df.groups:
            category, role, context_tag, response_tag = group
            if response_tag == 'offer':
                continue
            print '--------------------------------------'
            print 'category={}, role={}, context={}, response={}'.format(category, role, context_tag, response_tag)
            print '--------------------------------------'
            rows = [x[1] for x in df.get_group(group).iterrows()]
            rows = sorted(rows, key=lambda r: r['count'], reverse=True)
            for i, row in enumerate(rows):
                if i == n:
                    break
                print row['count'], row['response'].encode('utf-8')

class TemplateExtractor(object):
    stopwords = set(stopwords.words('english'))
    stopwords.update(['may', 'might', 'rent', 'new', 'brand', 'low', 'high', 'now', 'available'])

    def __init__(self, price_tracker):
        self.price_tracker = price_tracker
        self.ngram_counter = Trie()
        self.templates = []
        self.template_id = 0

    @classmethod
    def parse_utterance(cls, utterance):
        """Get speech acts of the utterance.
        """
        acts = []
        if SpeechActAnalyzer.is_question(utterance):
            acts.append('question')
        if utterance.prices:
            acts.append('price')
        if SpeechActAnalyzer.is_price(utterance):
            acts.append('vague-price')
        if SpeechActAnalyzer.is_agreement(utterance):
            acts.append('agree')
        if SpeechActAnalyzer.is_greeting(utterance):
            acts.append('greet')
        return acts

    @classmethod
    def parse_title(cls, tokens, kb):
        title = kb.facts['item']['Title'].lower().split()
        new_tokens = []
        for token in tokens:
            if token in title and not token in cls.stopwords:
                if len(new_tokens) > 0 and new_tokens[-1] == '{title}':
                    continue
                else:
                    new_tokens.append('{title}')
            else:
                new_tokens.append(token)
        return new_tokens

    @classmethod
    def parse_prices(cls, tokens, prev_prices, agent):
        new_tokens = []
        partner = 1 - agent
        price_templates = []
        for token in tokens:
            if is_entity(token):
                price = token.canonical.value
                if price == prev_prices['listing_price']:
                    token = '{listing_price}'
                elif price in prev_prices[agent]:
                    token = '{my_price}'
                elif price in prev_prices[partner]:
                    token = '{partner_price}'
                else:
                    token = '{price}'
                price_templates.append(token)
            new_tokens.append(token)
        return new_tokens, price_templates

    def add_template(self, category, role, context_tag, response_tag, response_tokens, context_tokens, n=4):
        """Add template and accumulate counts.

        Filter templates we are uncertain about and count ngrams in the template
        so that we can sample by frequency later.

        """
        if len(response_tokens) == 0:
            return
        num_prices = sum([1 if token == '{price}' else 0 for token in response_tokens])
        if num_prices > 1:
            return
        num_titles = sum([1 if token == '{title}' else 0 for token in response_tokens])
        if num_titles > 1:
            return

        # Count ngrams
        for ngram in self.ngrams(response_tokens, n):
            if ngram not in self.ngram_counter:
                self.ngram_counter[ngram] = 1
            else:
                self.ngram_counter[ngram] += 1

        row = {
                'category': category,
                'role': role,
                'response_tag': response_tag,
                'context_tag': context_tag,
                'response': response_tokens,
                'context': context_tokens,
                'id': self.template_id,
                }
        self.template_id += 1
        self.templates.append(row)

        return len(self.templates) - 1

    def parse_example(self, example, ngram_N):
        kbs = example.scenario.kbs
        category = kbs[0].category
        roles = {agent_id: kbs[agent_id].role for agent_id in (0, 1)}

        prices = {0: [], 1: [], 'listing_price': kbs[0].listing_price}
        mentioned_price = False
        prev_acts = []
        utterance_tags = ['<start>']
        prev_tokens = ['<start>']
        N = len(example.events)
        for i, event in enumerate(example.events):
            utterance = None
            if event.action == 'message':
                utterance = Utterance.from_text(event.data, self.price_tracker, kbs[event.agent])
            elif event.action == 'offer':
                if Dialogue.has_deal(example.outcome) and Dialogue.agreed_deal(example.outcome):
                    try:
                        utterance = Utterance(str(event.data['price']), [], event.action)
                    # Not sure why - sometimes offer event is None and is not the final outcome
                    except ValueError:
                        pass
            if utterance is None:
                continue

            agent = event.agent
            acts = self.parse_utterance(utterance)
            tokens, price_templates = self.parse_prices(utterance.tokens, prices, event.agent)
            tokens = self.parse_title(tokens, kbs[agent])

            if i == 0 and not 'price' in acts:
                tag = 'intro'
            elif 'greet' in acts and not 'price' in acts:
                tag = 'greet'
            elif 'price' in acts and not mentioned_price:
                mentioned_price = True
                tag = 'init-price'
            elif i + 1 < N and example.events[i+1].action == 'offer' and not '{price}' in price_templates:
                tag = 'agree'
            elif (not 'price' in acts) and 'vague-price' in acts:
                tag = 'vague-price'
            elif 'price' in acts and '{price}' in price_templates:
                tag = 'counter-price'
            elif acts == ['question']:
                tag = 'inquiry'
            elif not acts and prev_acts == ['question']:
                tag = 'inform'
            else:
                tag = 'unknown'

            # TODO:
            if event.action == 'offer':
                tokens = ['<offer>']
                tag = 'offer'

            template_id = self.add_template(category, roles[agent], utterance_tags[-1], tag, tokens, prev_tokens, n=ngram_N)
            event.template = template_id

            prices[agent] = [p.canonical.value for p in utterance.prices]
            prev_acts = acts
            prev_tokens = tokens
            utterance_tags.append(tag)

    def extract_templates(self, transcripts_paths, max_examples=-1, ngram_N=4, log=None):
        examples = read_examples(transcripts_paths, max_examples, Scenario)

        for example in examples:
            if Preprocessor.skip_example(example):
                continue
            self.parse_example(example, ngram_N)

        self.add_counts(ngram_N)
        self.detokenize_templates()

        if log:
            self.log_examples_with_templates(examples, log)

    def log_examples_with_templates(self, examples, log):
        for example in examples:
            if Preprocessor.skip_example(example):
                continue
            for event in example.events:
                template_id = event.template
                if template_id is not None:
                    event.template = self.templates[template_id]
        write_json([ex.to_dict() for ex in examples], log)

    def ngrams(self, tokens, n=1):
        for i in xrange(max(1, len(tokens)-n+1)):
            yield tokens[i:i+n]

    def detokenize_templates(self):
        #for k, temps in self.templates.iteritems():
        #    for temp in temps:
        for row in self.templates:
            row['response'] = detokenize(row['response'])
            row['context'] = detokenize(row['context'])

    def add_counts(self, n):
        for row in self.templates:
            tokens = row['response']
            counts = []
            for ngram in self.ngrams(tokens, n):
                counts.append(self.ngram_counter[ngram])
            if not counts:
                print tokens
                import sys; sys.exit()
            mean_count = np.mean(counts)
            row['count'] = mean_count


############# TEST #############
if __name__ == '__main__':
    import argparse
    from cocoa.core.util import write_pickle, read_pickle
    from core.price_tracker import PriceTracker

    parser = argparse.ArgumentParser()
    parser.add_argument('--transcripts', nargs='*', help='JSON transcripts to extract templates')
    parser.add_argument('--price-tracker-model')
    parser.add_argument('--max-examples', default=-1, type=int)
    parser.add_argument('--output', help='Path to save templates')
    parser.add_argument('--output-transcripts', help='Path to JSON examples with templates')
    parser.add_argument('--templates', help='Path to load templates')
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    if args.templates:
        templates = Templates.from_pickle(args.templates)
    else:
        price_tracker = PriceTracker(args.price_tracker_model)
        template_extractor = TemplateExtractor(price_tracker)
        template_extractor.extract_templates(args.transcripts, args.max_examples)
        write_pickle(template_extractor.templates, args.output)
        templates = Templates(template_extractor.templates)

    t = templates.templates
    response_tags = set(t.response_tag.values)
    tag_counts = []
    for tag in response_tags:
        tag_counts.append((tag, t[t.response_tag == tag].shape[0] / float(t.shape[0])))
    tag_counts = sorted(tag_counts, key=lambda x: x[1], reverse=True)
    for x in tag_counts:
        print x

    import sys; sys.exit()

    templates.dump(n=10)
    templates.build_tfidf()
    print templates.search('<start>', category='bike', role='seller')

    if args.debug:
        templates.dump()
        template = templates.choose(category='bike', role='buyer', response_tag='unknown')
        print template
