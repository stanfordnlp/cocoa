import matplotlib
matplotlib.use('Agg')
font_size = 18
matplotlib.rcParams.update({k: font_size for k in ('font.size', 'axes.labelsize', 'xtick.labelsize', 'ytick.labelsize', 'legend.fontsize')})
import matplotlib.pyplot as plt
import argparse
from src.basic.util import read_json, read_pickle
from itertools import izip
import numpy as np
import os
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--stats', nargs='+', help='Path to files containing the stats of a transcript')
parser.add_argument('--names', nargs='+', help='Names of systems corresponding to stats files')
parser.add_argument('--output', default='.', help='Path to output figure')
parser.add_argument('--completion', default=False, action='store_true', help='Plot completion')
parser.add_argument('--ngram-freqs', default=False, action='store_true', help='Plot ngram frequencies')
parser.add_argument('--utterance-freqs', default=False, action='store_true', help='Plot utterance frequencies')
args = parser.parse_args()

if args.ngram_freqs:
    stats = {}
    #stats_files = ['%s_ngram_counts.pkl' % x for x in args.stats]
    stats_files = args.stats
    for name, stats_file in izip(args.names, stats_files):
        stats[name] = read_pickle(stats_file)
    k = 10
    interval = 0.2
    bar_width = 0.2
    names = ['Human'] + [x for x in args.names if x != 'Human']
    colors = ['b', 'g', 'r', 'y', 'c', 'm'][:len(names)]
    for n in xrange(1, 4):
        plt.cla()
        ngrams = set()
        for name in names:
            sorted_words = sorted(stats[name][n].iteritems(), key=lambda x: x[1], reverse=True)
            ngrams.update([x[0] for x in sorted_words[:k]])
        ngrams = sorted(list(ngrams), key=lambda x: stats['Human'][n][x], reverse=True)[:15]
        label = [' '.join(x) if isinstance(x, tuple) else x for x in ngrams]
        pos = np.arange(len(ngrams))[::-1]
        for i, (name, color) in enumerate(izip(names, colors)):
            total = float(sum(stats[name][n].values()))
            counts = np.array([stats[name][n][word] / total for word in ngrams]) * 100.
            plt.barh(pos - i*bar_width, counts, height=bar_width, label=name, color=color)
        plt.yticks(pos - interval, label)
        plt.legend(loc='best')
        plt.xlabel('Percentage')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, '%d-gram.pdf' % n))

if args.utterance_freqs:
    stats = {}
    #stats_files = ['%s_utterance_counts.pkl' % x for x in args.stats]
    stats_files = args.stats
    for name, stats_file in izip(args.names, stats_files):
        bigram_counts = read_pickle(stats_file)
        stats[name] = defaultdict(int, {(k1, k2): v for k1, d in bigram_counts.iteritems() for k2, v in d.iteritems()})
    k = 10
    interval = 0.15
    bar_width = 0.15
    names = ['Human'] + [x for x in args.names if x != 'Human']
    colors = ['b', 'g', 'r', 'y', 'c', 'm'][:len(names)]
    for n in xrange(1, 2):
        plt.cla()
        ngrams = set()
        for name in names:
            sorted_words = sorted(stats[name].iteritems(), key=lambda x: x[1], reverse=True)
            ngrams.update([x[0] for x in sorted_words[:k]])
        ngrams = sorted(list(ngrams), key=lambda x: (x[0], stats['Human'][x]), reverse=True)[:15]
        def utterance_label_to_str(x):
            return '%s-%s' % ('|'.join(x[0]) if isinstance(x[0], tuple) else x[0], '|'.join([str(e) for e in x[1]]))
        label = [' , '.join([utterance_label_to_str(w) for w in x]) for x in ngrams]
        pos = np.arange(len(ngrams))[::-1]
        for i, (name, color) in enumerate(izip(names, colors)):
            d = stats[name]
            total = float(sum(d.values()))
            counts = np.array([d[word] / total for word in ngrams]) * 100.
            plt.barh(pos - i*bar_width, counts, height=bar_width, label=name, color=color)
        plt.yticks(pos - interval, label)
        plt.legend(loc='best')
        plt.xlabel('Percentage')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, '%d-utterance.pdf' % n))

if args.completion:
    styles = ['r-o', 'b->', 'g-*', 'c-s', 'k-d', 'y-<']
    assert len(args.stats) == len(args.names) and len(args.names) <= len(styles)
    styles = styles[:len(args.stats)]
    #stats_files = ['%s_stats.json' % x for x in args.stats]
    stats_files = args.stats

    for name, stat_file, style in izip(args.names, stats_files, styles):
        data = read_json(stat_file)['total']['turns_vs_completed']
        sorted_data = sorted([(float(t), float(c)) for t, c in data.iteritems() if float(t) > 2], key=lambda x: x[0])
        turns = [x[0] for x in sorted_data]
        completion_rate = np.cumsum([x[1] for x in sorted_data])
        plt.plot(turns, completion_rate, style, label=name)
    plt.xlabel('Number of turns')
    plt.ylabel('Cumulative completion rate')
    plt.legend(loc='best')
    plt.savefig(os.path.join(args.output, 'turns.pdf'))

    plt.cla()
    for name, stat_file, style in izip(args.names, args.stats, styles):
        data = read_json(stat_file)['total']['select_vs_completed']
        sorted_data = sorted([(float(t), float(c)) for t, c in data.iteritems()], key=lambda x: x[0])
        select = [x[0] for x in sorted_data]
        completion_rate = np.cumsum([x[1] for x in sorted_data])
        plt.plot(select, completion_rate, style, label=name)
    plt.xlabel('Number of selection')
    plt.ylabel('Cumulative completion rate')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'select.pdf'))
