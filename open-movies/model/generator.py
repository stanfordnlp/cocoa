from itertools import ifilter
import numpy as np
import json

from cocoa.model.generator import Templates as BaseTemplates, Generator as BaseGenerator

from core.tokenizer import detokenize

UNK = '<unk>'

class Generator(BaseGenerator):
    def __init__(self, templates):
        super(Generator, self).__init__(templates)
        #self.titles = list(set([x for x in self.templates.title.values if x != UNK]))
        #self.context_titles = list(set([x for x in self.templates.context_title.values if x != UNK]))

    def iter_titles(self):
        return ifilter(lambda x: x != UNK, self.templates.title.values)

    def iter_context_titles(self):
        d = self.templates.loc[self.templates.context_tag == 'ask-plot']
        return ifilter(lambda x: x != UNK, d.context_title.values)

    def get_filter(self, used_templates=None, title=None, context_title=None, context_tag=None, tag=None, **kwargs):
        locs = [super(Generator, self).get_filter(used_templates)]
        if tag:
            self._add_filter(locs, self.templates.tag == tag)
            print 'tag:', tag, np.sum(locs[-1])
        if title:
            self._add_filter(locs, self.templates.title == title)
            print 'title:', title, np.sum(locs[-1])
        if context_title:
            self._add_filter(locs, self.templates.context_title == context_title)
            print 'context title:', context_title, np.sum(locs[-1])
        if context_tag:
            self._add_filter(locs, self.templates.context_tag == context_tag)
            print 'context_tag:', context_tag, np.sum(locs[-1])
        return self._select_filter(locs)

class Templates(BaseTemplates):
    def ambiguous_template(self, template):
        return False

    def add_template(self, utterance, dialogue_state):
        if self.finalized:
            print 'Cannot add templates.'
            return
        if not utterance.template or self.ambiguous_template(utterance.template):
            return
        title = UNK
        if utterance.lf.intent != 'done' and utterance.lf.titles:
            title = utterance.lf.titles[0].canonical.value
        context_title = dialogue_state.curr_title or UNK
        row = {
                'title': title,
                'context_title': context_title,
                'tag': utterance.lf.intent,
                'template': detokenize(utterance.template),
                'context_tag': dialogue_state.partner_act,
                'context': detokenize(dialogue_state.partner_template),
                'id': self.template_id,
                }
        #print 'add template:'
        #print 'context:', row['context']
        #print 'template:', row['template']
        self.template_id += 1
        self.templates.append(row)

    def read_reviews(self, path):
        movies = json.load(open(path, "r"))
        for i, row in enumerate(movies):
            title = row['title']
            for type_, sentences in row['templates'].iteritems():
                if type_ == 'plot':
                    context_tag = 'ask-plot'
                else:
                    context_tag = UNK
                tag = 'inform'
                for s in sentences:
                    if len(s.split()) < 20:
                        r = {
                                'title': UNK,
                                'context_title': title,
                                'tag': tag,
                                'context_tag': context_tag,
                                'context': 'i you',
                                'template': s,
                                'id': self.template_id,
                            }
                        self.template_id += 1
                        self.templates.append(r)

    def dump(self, n=-1):
        #df = self.templates.groupby(['title', 'context_tag', 'tag'])
        df = self.templates.groupby(['tag'])
        for group in df.groups:
            #title, context_tag, response_tag = group
            response_tag = group
            if response_tag == 'done':
                continue
            print '='*40
            #print 'title={}, context={}, response={}'.format(title.encode('utf8'), context_tag, response_tag)
            print 'response={}'.format(response_tag)
            print '='*40
            rows = [x[1] for x in df.get_group(group).iterrows()]
            for i, row in enumerate(rows):
                if i == n:
                    break
                print row['template'].encode('utf-8')
