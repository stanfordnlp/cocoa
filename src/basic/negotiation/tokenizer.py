from nltk.tokenize import word_tokenize

def stick_dollar_sign(tokens):
    '''
    '$', '1000' -> '$1000'
    '''
    new_tokens = []
    for token in tokens:
        if len(new_tokens) > 0 and new_tokens[-1] == '$':
            new_tokens[-1] = '$%s' % token
        else:
            new_tokens.append(token)
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

def tokenize(utterance):
    '''
    'hi there!' => ['hi', 'there', '!']
    '''
    utterance = utterance.encode('utf-8').lower()
    tokens = word_tokenize(utterance.decode('utf-8'))
    tokens = stick_marker_sign(tokens)
    tokens = stick_dollar_sign(tokens)
    return tokens


