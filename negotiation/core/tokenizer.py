import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import re
import string

def is_number(s):
    if re.match(r'[.,0-9]+', s):
        return True
    else:
        return False

def stick_dollar_sign(tokens):
    '''
    '$', '1000' -> '$1000'
    '''
    new_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == '$':
            # $100
            if i < len(tokens) - 1 and is_number(tokens[i+1]):
                new_tokens.append(token + tokens[i+1])
                i += 2
            # 100$
            elif i > 0 and is_number(tokens[i-1]):
                new_tokens[-1] = new_tokens[-1] + token
                i += 1
            else:
                new_tokens.append(token)
                i += 1
        else:
            new_tokens.append(token)
            i += 1
    return new_tokens

def stick_marker_sign(tokens):
    '''
    Don't split on markers <>
    '<', 'x', '>' -> '<x>'
    '''
    new_tokens = []
    in_brackets = False
    for tok in tokens:
        if in_brackets:
            new_tokens[-1] = new_tokens[-1] + tok
        else:
            new_tokens.append(tok)
        if tok == '<':
            in_brackets = True
        if tok == '>':
            in_brackets = False
    return new_tokens

def tokenize(utterance, lowercase=True):
    '''
    'hi there!' => ['hi', 'there', '!']
    '''
    #utterance = utterance.encode('utf-8')
    if lowercase:
        utterance = utterance.lower()
    # NLTK would not tokenize "xx..", so normalize dots to "...".
    utterance = re.sub(r'\.{2,}', '...', utterance)
    # Remove some weird chars
    utterance = re.sub(r'\\|>|/', ' ', utterance)
    tokens = word_tokenize(utterance)
    #tokens = stick_marker_sign(tokens)
    tokens = stick_dollar_sign(tokens)
    return tokens

def detokenize(tokens):
    new_tokens = []
    for token in tokens:
        if (token in string.punctuation or "'" in token) and len(new_tokens) > 0:
            new_tokens[-1] += token
        elif token == 'na' and len(new_tokens) > 0 and new_tokens[-1] in ('gon', 'wan'):
            new_tokens[-1] += token
        else:
            new_tokens.append(token)
    return ' '.join(new_tokens)

# ========= TEST ===========
if __name__ == '__main__':
    print tokenize("i have 10,000$!..")
    print tokenize("i haven't $10,000")

