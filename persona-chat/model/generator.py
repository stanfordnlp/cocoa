import numpy as np
import json

from cocoa.model.generator import Templates as BaseTemplates, Generator as BaseGenerator
from core.tokenizer import detokenize

class Generator(BaseGenerator):
    def get_filter(self, used_templates=None, topic=None, **kwargs):
        locs = [super(Generator, self).get_filter(used_templates)]
        if topic:  # will not run, since there's no code for deciding topic
            self._add_filter(locs, self.templates.topic == topic)
        return self._select_filter(locs)

class Templates(BaseTemplates):
    def ambiguous_template(self, template):
        # Check if there is ambiguity, in which case we will discard the template.
        return False

    def add_template(self, utterance, dialogue_state):
        if self.finalized:
            print 'Cannot add templates.'
            return
        if utterance.ambiguous_template or self.ambiguous_template(utterance.template):
            return
        row = {
                'topic': utterance.lf.topic,
                'id': self.template_id,
                }
        self.template_id += 1
        self.templates.append(row)

    def dump(self, n=-1):
        df = self.templates.groupby(['topic'])
        for group in df.groups:
            response_tag = group
            if response_tag == 'done':
                continue
            print '='*40
            print 'response={}'.format(response_tag)
            print '='*40
            rows = [x[1] for x in df.get_group(group).iterrows()]
            for i, row in enumerate(rows):
                if i == n:
                    break
                print row['template'].encode('utf-8')
