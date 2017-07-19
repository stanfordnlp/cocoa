from tokenizer import tokenize
from src.basic.entity import CanonicalEntity, Entity, is_entity
import re
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem.porter import PorterStemmer
from src.basic.util import read_pickle, write_pickle

def add_slot_detector_arguments(parser):
    parser.add_argument('--slot-fillers', help='Path to slot fillers collected during training')

class SlotDetector(object):
    '''
    Given an utterance and the KB, detect KB-dependent text spans, similar to a Lexicon.
    '''
    def __init__(self, filler_path=None):
        self.stopwords = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        if filler_path:
            self.fillers = read_pickle(filler_path)
        else:
            self.fillers = None

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
        if stem:
            tokens = [self.stemmer.stem(tok) for tok in tokens]
        return tokens

    def get_context(self, kb, stem=False):
        title = self.process_info(kb.facts['item']['Title'], tag=False, stem=stem)
        description = self.process_info(' '.join(kb.facts['item']['Description']), tag=False, stem=stem)
        context = title
        context = [x for x in context if not x in self.stopwords]
        return context

    def detect_slots(self, tokens, kb=None, context=None):
        if not context:
            context = self.get_context(kb, stem=True)
        labels = []
        stems = [self.stemmer.stem(x) if not is_entity(x) else None for x in tokens]
        for tok in stems:
            labels.append(1 if (tok is not None and tok in context) else 0)
        spans = self.label_to_span(labels)
        slots = []
        for span in spans:
            if not isinstance(span, tuple):
                slots.append(tokens[span])
            else:
                slots.append(Entity(surface=' '.join(tokens[span[0]: span[1]]),
                    canonical=CanonicalEntity(value='', type='slot')))
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

    def context_to_fillers(self, kb):
        tokens = self.get_context(kb, stem=False)
        fillers = []
        for n in (1, 2):
            fillers.extend(self.get_ngram_match(tokens, n))
        return fillers

    def collect_fillers(self, examples, preprocessor, output, verbose=False):
        fillers = set()
        for ex in examples:
            if preprocessor.skip_example(ex):
                continue
            kb = ex.scenario.kbs[1]
            if verbose:
                kb.dump()
            kb_context = self.get_context(kb)
            for e in ex.events:
                kb = ex.scenario.kbs[e.agent]
                utterance = preprocessor.process_event(e, e.agent, kb)
                utterance = self.detect_slots(utterance, context=kb_context)
                if verbose:
                    print ' '.join(['[%s]' % x.surface if is_entity(x) else x for x in utterance])
                fillers.update([x.surface for x in utterance if is_entity(x) and x.canonical.type == 'slot'])
        write_pickle(fillers, output)


if __name__ == '__main__':
    from src.basic.negotiation.price_tracker import PriceTracker, add_price_tracker_arguments
    from src.basic.schema import Schema
    from src.basic.dataset import read_examples
    from src.model.negotiation.preprocess import Preprocessor, markers
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--schema-path', help='Input path that describes the schema of the domain', required=True)
    parser.add_argument('--transcripts', nargs='+', default=[])
    parser.add_argument('--max-examples', type=int, help='Maximum number of examples to read from transcripts')
    parser.add_argument('--output', help='Path to output model')
    parser.add_argument('--verbose', action='store_true')
    add_price_tracker_arguments(parser)
    args = parser.parse_args()

    #examples = read_examples(None, args.transcripts, args.max_examples)
    #slot_detector = SlotDetector('/scr/hehe/game-dialogue/slot_fillers.pkl')
    #for ex in examples:
    #    kb = ex.scenario.kbs[0]
    #    print kb.facts['item']['Title']
    #    print slot_detector.context_to_fillers(kb)
    #import sys; sys.exit()

    lexicon = PriceTracker(args.price_tracker_model)
    slot_detector = SlotDetector()
    schema = Schema(args.schema_path)
    preprocessor = Preprocessor(schema, lexicon, 'canonical', 'canonical', 'canonical')

    examples = read_examples(None, args.transcripts, args.max_examples)
    slot_detector.collect_fillers(examples, preprocessor, args.output, args.verbose)

    #print SlotDetector.label_to_span([0,0,1,1,0,1])


