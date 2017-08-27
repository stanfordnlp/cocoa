import argparse
from src.basic.util import read_json, write_json
import numpy as np
from itertools import izip

parser = argparse.ArgumentParser()
parser.add_argument('--example-paths', help='Transciprts paths', nargs='*', default=[])
parser.add_argument('--train-frac', help='Fraction of training examples', type=float, default=0.6)
parser.add_argument('--test-frac', help='Fraction of test examples', type=float, default=0.2)
parser.add_argument('--dev-frac', help='Fraction of dev examples', type=float, default=0.2)
parser.add_argument('--output-path', help='Output path for splits')
args = parser.parse_args()

np.random.seed(0)
json_data = ([], [], [])
for path in args.example_paths:
    examples = read_json(path)
    folds = np.random.choice(3, len(examples), p=[args.train_frac, args.dev_frac, args.test_frac])
    for ex, fold in izip(examples, folds):
        json_data[fold].append(ex)

for fold, dataset in izip(('train', 'dev', 'test'), json_data):
    if len(dataset) > 0:
        write_json(dataset, '%s%s.json' % (args.output_path, fold))
