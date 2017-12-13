from collections import defaultdict
import numpy as np
from cocoa.model.generator import Templates as BaseTemplates, Generator as BaseGenerator
from core.tokenizer import detokenize

class Generator(BaseGenerator):
    def get_filter(self, used_templates=None, signature=None, context_tag=None, tag=None, **kwargs):
        locs = [super(Generator, self).get_filter(used_templates)]
        if signature:
            self._add_filter(locs, self.templates.signature == signature)
            # signature must be satisfied
            if np.sum(locs[-1]) == 0:
                print 'no signature=', signature
                return None
        if tag:
            print 'tag=', tag
            self._add_filter(locs, self.templates.tag == tag)
        if context_tag:
            self._add_filter(locs, self.templates.context_tag == context_tag)
        return self._select_filter(locs)

class Templates(BaseTemplates):
    def _get_entities(self, template):
        entities = defaultdict(int)
        for token in template:
            if token == '{number}':
                entities[token] += 1
            elif token[0] == '{' and token[-1] == '}':
                # E.g. {hobby[0]}
                entities[token[1:-4]] += 1
        return entities

    def ambiguous_template(self, template):
        """Check if there is ambiguity, in which case we will discard the template.
        """
        entities = self._get_entities(template)
        numbers = entities['{number}']
        if numbers > 0 and not (numbers == 1 and len(entities) == 2 and sum(entities.values()) == 2):
            return True
        return False

    def add_template(self, utterance, dialogue_state):
        if self.finalized:
            print 'Cannot add templates.'
            return
        if utterance.ambiguous_template or self.ambiguous_template(utterance.template):
            return

        row = {
                'tag': utterance.lf.intent,
                'template': detokenize(utterance.template),
                'signature': None if utterance.lf.intent == 'select' else utterance.lf.signature,
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
        df = self.templates.groupby(['signature', 'context_tag', 'tag'])
        for group in df.groups:
            signature, context_tag, response_tag = group
            if response_tag == 'select':
                continue
            print '='*40
            print 'signature={}, context={}, response={}'.format(signature, context_tag, response_tag)
            print '='*40
            rows = [x[1] for x in df.get_group(group).iterrows()]
            #rows = sorted(rows, key=lambda r: r['count'], reverse=True)
            for i, row in enumerate(rows):
                if i == n:
                    break
                print row['template'].encode('utf-8')
