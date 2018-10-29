from tokenizer import tokenize
from cocoa.core.entity import CanonicalEntity, Entity, is_entity
import re
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk import FreqDist
import numpy as np
from collections import defaultdict
from cocoa.core.util import write_json, read_json
from nltk.stem.porter import PorterStemmer

def add_slot_detector_arguments(parser):
    parser.add_argument('--slot-scores', help='Path to slot scores computed during training')
    #parser.add_argument('--stop-words', default='data/common_words.txt', help='Path to list of stop words')

class SlotDetector(object):
    '''
    Given an utterance and the KB, detect KB-dependent text spans, similar to a Lexicon.
    '''
    def __init__(self, slot_scores_path=None, stop_words_path=None, threshold=4.):
        if not stop_words_path:
            self.stopwords = set(stopwords.words('english'))
        else:
            with open(stop_words_path, 'r') as fin:
                self.stopwords = set(fin.read().split()[:200])
        self.stopwords.update(['.', '...', ',', '?', '!', '"', "n't", "'m", "'d", "'ll"])
        if slot_scores_path:
            self.slot_scores = read_json(slot_scores_path)
        self.threshold = threshold
        self.stemmer = PorterStemmer()

    @classmethod
    def label_to_span(cls, labels):
        '''
        [0, 0, 1, 1] -> [0, 1, (2, 4)]
        '''
        spans = []
        start, end = None, None
        for i, l in enumerate(labels):
            if l == 1:
                if start is None:
                    start, end = i, i + 1
                else:
                    end += 1
            else:
                if start is not None:
                    spans.append((start, end))
                    start, end = None, None
                spans.append(i)
        if start is not None:
            spans.append((start, end))
        return spans

    def process_info(self, s, tag=False, stem=False):
        tokens = tokenize(re.sub(r'[^\w0-9]', ' ', s))
        if tag:
            tokens_pos = pos_tag(tokens)
            tokens = [w for w, t in tokens_pos if t.startswith('NN')]
        return tokens

    def _get_slot_labels(self, role, category, context_tokens, dialogue_tokens, stem=False):
        labels = []
        slot_scores = self.slot_scores[role][category]
        for tok in dialogue_tokens:
            label = None
            if self._possible_slot_word(tok):
                if stem:
                    tok = self.stemmer.stem(tok)
                if tok in context_tokens:
                    label = 1
                elif tok in slot_scores:
                    scores = slot_scores[tok]
                    for ctok in context_tokens:
                        if (ctok in scores) and (scores[ctok] > self.threshold):
                            label = 1
                            break
            labels.append(label if label is not None else 0)
        return labels

    def detect_slots(self, tokens, kb, context=None, join=False, stem=False):
        '''
        join: join consecutive slot words to one entity
        '''
        if not context:
            context = self.get_context_tokens((kb,), stem=stem)
        #print 'context tokens:', context

        role = kb.facts['personal']['Role']
        category = kb.facts['item']['Category']
        labels = self._get_slot_labels(role, category, context, tokens, stem=stem)

        slot_entity = lambda s: Entity(surface=s,
                    canonical=CanonicalEntity(value='', type='slot'))
        if not join:
            slots = [slot_entity(x) if labels[i] == 1 else x for i, x in enumerate(tokens)]
        else:
            spans = self.label_to_span(labels)
            slots = [tokens[x] if not isinstance(x, tuple) else
                    slot_entity(' '.join(tokens[x[0]: x[1]]))
                    for x in spans]
        return slots

    def ngrams(self, tokens, n=1):
        for i in xrange(len(tokens)-n+1):
            yield tokens[i:i+n]

    # TODO: skip gram
    def get_ngram_match(self, tokens, n=1):
        matches = []
        for phrase in self.ngrams(tokens, n):
            phrase = ' '.join(phrase)
            if phrase in self.fillers:
                matches.append(phrase)
        return matches

    def get_dialogue_tokens(self, ex, preprocessor, stem=False):
        tokens = {'buyer': [], 'seller': []}
        kbs = ex.scenario.kbs
        for e in ex.events:
            kb = kbs[e.agent]
            role = kb.facts['personal']['Role']
            utterance = preprocessor.process_event(e, e.agent, kb)
            toks = self._filter_words(utterance)
            if stem:
                toks = [self.stemmer.stem(tok.decode('utf-8')) for tok in toks]
            tokens[role].extend(toks)
        return tokens

    def _process_context(self, s):
        '''
        Remove special chars in title and description.
        '''
        tokens = tokenize(re.sub(r'[^\w0-9]', ' ', s))
        return tokens

    def _possible_slot_word(self, tok):
        return not (tok in self.stopwords or is_entity(tok))

    def _filter_words(self, words):
        '''
        Remove words that are not likely to be slot words.
        '''
        tokens = [x for x in words if self._possible_slot_word(x)]
        return tokens

    def get_context_tokens(self, kbs, stem=False):
        if not (isinstance(kbs, list) or isinstance(kbs, tuple)):
            kbs = (kbs,)
        tokens = {}
        title = self._process_context(kbs[0].facts['item']['Title'])
        title = self._filter_words(title)
        for kb in kbs:
            role = kb.facts['personal']['Role']
            description = self._process_context(' '.join(kb.facts['item']['Description']))
            description = self._filter_words(description)
            context = title + description
            if stem:
                context = [self.stemmer.stem(tok) for tok in context]
            tokens[role] = context
        if len(kbs) == 1:
            return tokens[role]
        return tokens

    def get_collocate_data(self, examples, preprocessor, stem=False):
        '''
        Process context (KB) text and dialoge text.
        '''
        collocate_data = []
        for ex in examples:
            if preprocessor.skip_example(ex):
                continue
            kbs = ex.scenario.kbs
            category = kbs[0].facts['item']['Category']
            dialogue_tokens = self.get_dialogue_tokens(ex, preprocessor, stem=stem)
            context_tokens = self.get_context_tokens(kbs, stem=stem)
            for role in ('buyer', 'seller'):
                collocate_data.append({
                    'category': category,
                    'role': role,
                    'dialogue_tokens': dialogue_tokens[role],
                    'context_tokens': context_tokens[role],
                    })
        return collocate_data

    def compute_scores(self, collocate_data, role=None, category=None):
        data = collocate_data
        if role:
            data = [x for x in collocate_data if x['role'] == role]
        if category:
            data = [x for x in collocate_data if x['category'] == category]

        dialogue_fd = FreqDist([w for d in data for w in d['dialogue_tokens']])
        context_fd = FreqDist([w for d in data for w in d['context_tokens']])
        collocate_fd = FreqDist([(w_c, w_d) for d in data for w_c in d['context_tokens'] for w_d in d['dialogue_tokens']])

        N_d = dialogue_fd.N()
        N_c = context_fd.N()
        N_cd = collocate_fd.N()

        scores = defaultdict(dict)
        const = np.log(N_c) + np.log(N_d) - np.log(N_cd)
        for pair in collocate_fd:
            w_c, w_d = pair
            s = np.log(collocate_fd[pair]) - np.log(dialogue_fd[w_d]) - np.log(context_fd[w_c]) + const
            scores[w_d][w_c] = s

        return dict(scores)

    def train(self, examples, preprocessor, output_path, stem=False):
        d = self.get_collocate_data(examples, preprocessor, stem=stem)
        scores = defaultdict(dict)
        for role in ('seller', 'buyer'):
            for category in ('car', 'housing', 'phone', 'electronics', 'furniture', 'bike'):
                scores[role][category] = self.compute_scores(d, role=role, category=category)
        write_json(dict(scores), output_path)
        return scores

if __name__ == '__main__':
    from cocoa.core.negotiation.price_tracker import PriceTracker, add_price_tracker_arguments
    from cocoa.core.schema import Schema
    from cocoa.core.dataset import read_examples
    from cocoa.model.negotiation.preprocess import Preprocessor, markers
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--schema-path', help='Input path that describes the schema of the domain', required=True)
    parser.add_argument('--transcripts', nargs='+', default=[])
    parser.add_argument('--max-examples', type=int, help='Maximum number of examples to read from transcripts')
    parser.add_argument('--output', help='Path to output model')
    parser.add_argument('--test', action='store_true')
    add_price_tracker_arguments(parser)
    add_slot_detector_arguments(parser)
    args = parser.parse_args()

    lexicon = PriceTracker(args.price_tracker_model)
    schema = Schema(args.schema_path)
    preprocessor = Preprocessor(schema, lexicon, 'canonical', 'canonical', 'canonical')
    examples = read_examples(None, args.transcripts, args.max_examples)

    if args.test:
        slot_detector = SlotDetector(slot_scores_path=args.slot_scores)
        print sorted(slot_detector.slot_scores['seller']['furniture']['hi'].items(), key=lambda x: x[1], reverse=True)[:10]
        #import sys; sys.exit()
        for ex in examples:
            kbs = ex.scenario.kbs
            kbs[1].dump()
            for e in ex.events:
                kb = kbs[e.agent]
                utterance = preprocessor.process_event(e, e.agent, kb)
                print ' '.join([str(x) for x in slot_detector.detect_slots(utterance, kb, stem=True)])

    else:
        slot_detector = SlotDetector()
        scores = slot_detector.train(examples, preprocessor, args.output, stem=True)


