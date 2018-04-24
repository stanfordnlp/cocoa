#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import six
import sys
import os
import numpy as np
import argparse
import torch
from tqdm import tqdm as progress_bar

from cocoa.io.utils import read_pickle

parser = argparse.ArgumentParser(description='embeddings_to_torch.py')
parser.add_argument('--emb-file', required=True,
                help="Embeddings from this file, will often be Glove or Word2Vec")
parser.add_argument('--output-file', required=True,
                help="Pytorch embeddings into this file for the prepared data")
parser.add_argument('--vocab-file', required=True,
                help="Dictionary that maps a word to its embedding index")
parser.add_argument('--vocab-type', default='utterance',
                choices=['utterance', 'kb', 'cat', 'lf'],
                help='type of mapping being embedded into pytorch')
parser.add_argument('--verbose', action="store_true", default=False)
opt = parser.parse_args()


def get_vocabs(vocab_path, vocab_type):
    mappings = read_pickle(vocab_path)
    mapping_key = "{}_vocab".format(vocab_type)
    vocab = mappings[mapping_key]
    print('{0} vocab size: {1}'.format(vocab_type, len(vocab)) )
    return vocab

def get_embeddings(file_):
    embs = dict()
    with open(file_, 'r') as f:
        for l in progress_bar(f):
            l_split = l.strip().split()
            embs[l_split[0]] = [float(em) for em in l_split[1:]]
    print("Got {} embeddings from {}".format(len(embs), file_))

    return embs


def match_embeddings(vocab, emb):
    dim = len(six.next(six.itervalues(emb)))
    filtered_embeddings = np.zeros((len(vocab), dim))
    count = {"match": 0, "miss": 0}
    for w, w_id in vocab.word_to_ind.items():
        if w in emb:
            filtered_embeddings[w_id] = emb[w]
            count['match'] += 1
        else:
            if opt.verbose:
                print(u"not found:\t{}".format(w), file=sys.stderr)
            count['miss'] += 1

    return torch.Tensor(filtered_embeddings), count


def main():
    vocab = get_vocabs(opt.vocab_file, opt.vocab_type)
    embeddings = get_embeddings(opt.emb_file)

    filtered_embeddings, count = match_embeddings(vocab, embeddings)

    print("\nMatching: ")
    match_percent = [_['match'] / (_['match'] + _['miss']) * 100
                     for _ in [count,]]
    print("\t* vocab: %d match, %d missing, (%.2f%%)" % (count['match'],
                                                       count['miss'],
                                                       match_percent[0]))

    print("\nFiltered embeddings:")
    print("\t* vocab: ", filtered_embeddings.size())

    output_file = os.path.join(opt.output_file, opt.vocab_type + "_glove.pt")
    print("\nSaving embedding as:\n\t%s"
          % (output_file,))
    torch.save(filtered_embeddings, output_file)
    print("\nDone.")


if __name__ == "__main__":
    main()
