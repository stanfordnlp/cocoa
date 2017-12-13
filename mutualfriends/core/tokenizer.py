import re
from cocoa.core.tokenizer import detokenize

def tokenize(utterance):
    '''
    'hi there!' => ['hi', 'there', '!']
    '''
    utterance = utterance.encode('utf-8').lower()
    # Remove '-' to match lexicon preprocess
    for s in (' - ', '-'):
        utterance = utterance.replace(s, ' ')
    # Split on punctuation
    tokens = re.findall(r"[\w']+|[.,!?;&-]", utterance)
    return tokens

