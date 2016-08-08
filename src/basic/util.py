import random
import json
import string

def random_multinomial(probs):
    target = random.random()
    i = 0
    accum = 0
    while True:
        accum += probs[i]
        if accum >= target:
            return i
        i += 1

def generate_uuid(prefix):
    return prefix + '_' + ''.join([random.choice(string.digits + string.letters) for _ in range(16)])

def read_json(path):
    try:
        return json.load(open(path))
    except:
        raise Exception('Error reading JSON from %s' % path)

def write_json(raw, path):
    with open(path, 'w') as out:
        print >>out, json.dumps(raw)
