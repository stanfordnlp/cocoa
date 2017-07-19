import argparse
from src.basic.util import read_json, write_json

parser = argparse.ArgumentParser()
parser.add_argument('--input')
parser.add_argument('--output')
parser.add_argument('-n', type=int, default=2, help='Number of partitions')
parser.add_argument('--split', action='store_true', help='Split inputs')
parser.add_argument('--merge', action='store_true', help='Merge inputs')
args = parser.parse_args()

def chunks(l, N):
    '''
    Split the list l into N chunks.
    '''
    num_chunks = 0
    n = len(l) / N
    for i in range(0, len(l), n):
        num_chunks += 1
        if num_chunks == N:
            yield l[i:]
            break
        else:
            yield l[i:i + n]

if args.split:
    examples = read_json(args.input)
    for i, ex in enumerate(chunks(examples, args.n)):
        print 'write to', '%s.%d' % (args.output, i), len(ex)
        write_json(ex, '%s.%d' % (args.output, i))
elif args.merge:
    examples = []
    for i in xrange(args.n):
        filename = '%s.%d' % (args.input, i)
        ex = read_json(filename)
        examples.extend(ex)
    write_json(examples, args.output)
