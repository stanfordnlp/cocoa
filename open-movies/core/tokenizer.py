from nltk.tokenize import word_tokenize
import re

def tokenize(utterance, lowercase=True):
    '''
    'hi there!' => ['hi', 'there', '!']
    '''
    utterance = utterance.encode('utf-8')
    if lowercase:
        utterance = utterance.lower()
    # NLTK would not tokenize "xx..", so normalize dots to "...".
    utterance = re.sub(r'\.{2,}', '...', utterance)
    # Remove some weird chars
    utterance = re.sub(r'\\|>|/', ' ', utterance)
    tokens = word_tokenize(utterance)
    return tokens

# ========= TEST ===========
if __name__ == '__main__':
    print tokenize("i have 10,000$!..")
    print tokenize("i haven't $10,000")

