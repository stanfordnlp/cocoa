from nltk.tokenize.moses import MosesDetokenizer

detokenizer = MosesDetokenizer()

def detokenize(tokens):
    return detokenizer.detokenize(tokens, return_str=True)
