import string

from cocoa.turk.eval_data import EvalData as BaseEvalData, add_eval_data_arguments

class EvalData(BaseEvalData):
    @classmethod
    def get_agent_name(cls, turns):
        names = []
        for utterance in turns:
            if utterance[0] == '<go-b>':
                role = 'buyer'
            elif utterance[0] == '<go-s>':
                role = 'seller'
            elif len(names) > 0:
                role = 'buyer' if names[-1] == 'seller' else 'seller'
            else:
                raise Exception('Cannot infer agent name')
            names.append(role)
        return names

    @classmethod
    def process_utterance(cls, utterance, role=''):
        tokens = []
        for w in utterance:
            if not isinstance(w, basestring):
                if w[1][1] == 'price' or w[1] == 'price':
                    tokens.append('PRICE')
                else:
                    raise ValueError
            elif w in ('<done>', '<quit>'):
                tokens.append(w[1:-1].upper())
            # Category markers
            elif len(w) > 2 and w[0] == '<' and w[-1] == '>':
                continue
            # De-tokenize
            elif (w in string.punctuation or "'" in w) and len(tokens) > 0:
                tokens[-1] += w
            else:
                tokens.append(w)
        return super(EvalData, cls).process_utterance(' '.join(tokens), role)

    @classmethod
    def valid_example(cls, example, num_context_utterances):
        last_utterance = example['prev_turns'][-1]
        if '<offer>' in last_utterance or '<accept>' in last_utterance or '<reject>' in last_utterance:
            return False
        return super(EvalData, cls).valid_example(example, num_context_utterances)
