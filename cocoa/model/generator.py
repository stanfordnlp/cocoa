import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from cocoa.core.util import read_pickle, write_pickle
from cocoa.model.counter import build_vocabulary, count_ngrams
from cocoa.model.ngram import MLENgramModel

from core.tokenizer import detokenize

class Generator(object):
    def __init__(self, templates):
        self.templates = templates.templates
        self.vectorizer = TfidfVectorizer()
        self.build_tfidf()

    def build_tfidf(self):
        documents = self.templates['context'].values
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

    def _add_filter(self, locs, cond):
        locs.append(locs[-1] & cond)

    def _select_filter(self, locs):
        print [np.sum(loc) for loc in locs]
        for loc in locs[::-1]:
            if np.sum(loc) > 0:
                return loc
        return locs[0]

    def get_filter(self, used_templates=None):
        if used_templates:
            loc = (~self.templates.id.isin(used_templates))
            if np.sum(loc) > 0:
                return loc
        # All templates
        return self.templates.id.notnull()

    def retrieve(self, context, used_templates=None, topk=20, T=1., **kwargs):
        loc = self.get_filter(used_templates=used_templates, **kwargs)
        if loc is None:
            return None

        if isinstance(context, list):
            context = detokenize(context)
        features = self.vectorizer.transform([context])
        scores = self.tfidf_matrix * features.T
        scores = scores.todense()[loc]
        scores = np.squeeze(np.array(scores), axis=1)
        ids = np.argsort(scores)[::-1][:topk]

        candidates = self.templates[loc]
        candidates = candidates.iloc[ids]
        rows = self.templates[loc]
        rows = rows.iloc[ids]
        logp = rows['logp'].values

        return self.sample(logp, candidates, T)

    def sample(self, scores, templates, T=1.):
        probs = self.softmax(scores, T=T)
        template_id = np.random.multinomial(1, probs).argmax()
        template = templates.iloc[template_id]
        return template

    def softmax(self, scores, T=1.):
        exp_scores = np.exp(scores / T)
        return exp_scores / np.sum(exp_scores)


class Templates(object):
    """Data structure for templates.
    """
    def __init__(self, templates=[], finalized=False):
        self.templates = templates
        self.template_id = len(templates)
        self.finalized = finalized

    @classmethod
    def from_pickle(cls, path):
        templates = read_pickle(path)
        return cls(templates=templates, finalized=True)

    def add_template(self, utterance, dialogue_state):
        raise NotImplementedError

    def finalize(self):
        self.templates = pd.DataFrame(self.templates)
        self.score_templates()
        self.finalized = True

    def save(self, output):
        assert self.finalized
        write_pickle(self.templates, output)

    def score_templates(self):
        sequences = [s.split() for s in self.templates.template.values]
        vocab = build_vocabulary(1, *sequences)
        counter = count_ngrams(3, vocab, sequences, pad_left=True, pad_right=False)
        model = MLENgramModel(counter)
        scores = [-1.*model.entropy(s)*len(s) for s in sequences]
        if not 'logp' in self.templates.columns:
            self.templates.insert(0, 'logp', 0)
        self.templates['logp'] = scores
