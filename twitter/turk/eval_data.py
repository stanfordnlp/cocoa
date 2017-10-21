import string

from cocoa.turk.eval_data import EvalData as BaseEvalData, add_eval_data_arguments

class EvalData(BaseEvalData):
    @classmethod
    def get_agent_name(cls, turns):
        names = []
        for i, utterance in enumerate(turns):
            if i % 2 == 0:
                names.append('A')
            else:
                names.append('B')
        return names

    @classmethod
    def valid_example(cls, example, num_context_utterances):
        return True

    @classmethod
    def process_utterance(cls, utterance, role=''):
        utterance = utterance.replace('< url >', 'URL')
        utterance = utterance.replace('<UNK>', 'UNK')
        return super(EvalData, cls).process_utterance(utterance, role)
