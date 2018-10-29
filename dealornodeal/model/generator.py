import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from cocoa.model.generator import Templates as BaseTemplates, Generator as BaseGenerator

from core.tokenizer import detokenize
from parser import Parser

class Generator(BaseGenerator):
    def get_filter(self, used_templates=None, proposal_type=None, context_tag=None, tag=None, **kwargs):
        print 'filter:', proposal_type, context_tag, tag
        locs = [super(Generator, self).get_filter(used_templates)]
        if proposal_type:
            self._add_filter(locs, self.templates.proposal_type == proposal_type)
            # proposal_type must be satisfied
            if np.sum(locs[-1]) == 0:
                return None
        if tag:
            self._add_filter(locs, self.templates.tag == tag)
        if context_tag:
            self._add_filter(locs, self.templates.context_tag == context_tag)
        return self._select_filter(locs)

class Templates(BaseTemplates):
    def ambiguous_template(self, template):
        """Check if there is ambiguity, in which case we will discard the template.
        """
        if len(template) == 0 or '{number}' in template:
            return True
        num_item = 0
        for i, token in enumerate(template):
            if token in ('{book}', '{hat}', '{ball}'):
                item = token[1:-1]
                num_item += 1
                if not (i > 1 and template[i-1] in ('{{{0}-number}}'.format(item), 'the', 'all')):
                    return True
        if num_item > 3:
            return True
        return False

    def add_template(self, utterance, dialogue_state):
        if self.finalized:
            print 'Cannot add templates.'
            return
        if utterance.ambiguous_template or self.ambiguous_template(utterance.template):
            return
        proposal_type = utterance.lf.proposal_type if utterance.lf.intent == 'propose' else 'none'
        row = {
                'tag': utterance.lf.intent,
                'template': detokenize(utterance.template),
                'proposal_type': proposal_type,
                'context_tag': dialogue_state.partner_act,
                'context': detokenize(dialogue_state.partner_template),
                'id': self.template_id,
                }
        #print 'add template:'
        #print 'context:', row['context']
        #print 'template:', row['template']
        self.template_id += 1
        self.templates.append(row)

    def dump(self, n=-1):
        df = self.templates.groupby(['proposal_type', 'context_tag', 'tag'])
        for group in df.groups:
            proposal_type, context_tag, response_tag = group
            if response_tag in ('reject', 'select'):
                continue
            print '='*40
            print 'proposal_type={}, context={}, response={}'.format(proposal_type, context_tag, response_tag)
            print '='*40
            rows = [x[1] for x in df.get_group(group).iterrows()]
            #rows = sorted(rows, key=lambda r: r['count'], reverse=True)
            for i, row in enumerate(rows):
                if i == n:
                    break
                print row['template'].encode('utf-8')
