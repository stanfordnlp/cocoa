"""Basic IO utils.
"""

import os
import ujson as json
import cPickle as pickle

def create_path(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

def read_json(path):
    try:
        return json.load(open(path))
    except:
        raise Exception('Error reading JSON from %s' % path)

def write_json(raw, path, ensure_path=False):
    if ensure_path:
        create_path(path)
    with open(path, 'w') as out:
        print >>out, json.dumps(raw)

def read_pickle(path):
    with open(path, 'rb') as fin:
        return pickle.load(fin)

def write_pickle(obj, path, ensure_path=False):
    if ensure_path:
        create_path(path)
    with open(path, 'wb') as fout:
        pickle.dump(obj, fout)

