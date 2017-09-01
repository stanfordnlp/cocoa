import os.path
import csv
import sqlite3
import json
from collections import defaultdict
from itertools import izip
import numpy as np
from argparse import ArgumentParser
from boto.mturk.connection import MTurkConnection
from boto.mturk.price import Price
from boto.mturk.connection import MTurkRequestError

from cocoa.analysis.utils import get_turns_per_agent, get_avg_tokens_per_agent
from cocoa.core.util import read_json, write_json, normalize
# TODO
from cocoa.web.main.db_reader import read_results_csv, chat_to_worker_id
from cocoa.lib.logstats import update_summary_map

def get_winner(transcript):
    scenario = transcript["scenario"]
    kbs = scenario["kbs"]
    try:
        offer = transcript["outcome"]["offer"]
        if isinstance(offer, dict):
            offer = offer['price']
        if offer is None:
            return -1
    except KeyError:
        return -1
    # TODO: once we finalized the scenarios we should just have one case
    refs = {0: kbs[0]['personal']['Bottomline'],
            1: kbs[1]['personal']['Bottomline']}
    ref_price = 'bottomline'
    if refs[0] is None or refs[1] is None:
        refs = {0: kbs[0]['personal']['Target'],
                1: kbs[1]['personal']['Target']}
        ref_price = 'target'


    roles = {kbs[0]['personal']['Role']: 0,
             kbs[1]['personal']['Role']: 1}

    diffs = {}

    # for seller, offer is probably lower than target
    seller_idx = roles['seller']
    diffs[seller_idx] = abs(offer - refs[seller_idx])

    # for buyer, offer is probably
    buyer_idx = roles['buyer']
    diffs[buyer_idx] = abs(refs[buyer_idx] - offer)

    if ref_price == 'bottomline':
        # Winner if farther from bottomline
        if diffs[0] < diffs[1]:
            return 1
        elif diffs[1] < diffs[0]:
            return 0
    else:
        # Winner is closer to target
        if diffs[0] < diffs[1]:
            return 0
        elif diffs[1] < diffs[0]:
            return 1

    return -1

def get_win_rate_per_agent(chat):
    winner = get_winner(chat)
    score = {0: 0, 1: 0}
    if winner != -1:
        score[winner] = 1
    return score

def get_stats(chat, survey):
    stats = {0: {}, 1: {}}
    tokens = get_avg_tokens_per_agent(chat)
    turns = get_turns_per_agent(chat)
    win_rates = get_win_rate_per_agent(chat)
    for name, data in izip(('num_tokens', 'num_turns', 'win_rate'), (tokens, turns, win_rates)):
        for i in xrange(2):
            stats[i][name] = data[i]
    for q in ('fluent', 'persuasive', 'negotiator'):
        for i in xrange(2):
            s = survey[str(i)]
            if q not in s:
                stats[i][q] = 3
            else:
                stats[i][q] = s[q]
    return stats

def is_valid(agent_stats):
    '''
    Don't count chats where there is only one person talking.
    '''
    if agent_stats[1]['num_turns'] < 3 or agent_stats[0]['num_turns'] < 3:
        return False
    return True

def update_stats(chat_workers, chats, surveys, worker_stats, worker_chats):
    workers = set()
    for chat_id, agent_wids in chat_workers.iteritems():
        try:
            chat = chats[chat_id]
            survey = surveys[chat_id]
        except KeyError:
            continue
        agent_stats = get_stats(chat, survey)
        if is_valid(agent_stats):
            for agent_id, agent_wid in agent_wids.iteritems():
                if agent_wid is not None:
                    workers.add(agent_wid)
                    update_summary_map(worker_stats[agent_wid], agent_stats[int(agent_id)])
                    worker_chats[agent_wid].append((agent_id, chat))
    print 'Update for %d workers' % len(workers)

def update_worker_stats(result_path, worker_stats, worker_chats):
    db_path = os.path.join(result_path, 'chat_state.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    transcripts = os.path.join(result_path, 'transcripts/transcripts.json')
    chats = read_json(transcripts)
    chats = {chat['uuid']: chat for chat in chats}
    print 'Number of chats:', len(chats)

    survey_path = os.path.join(result_path, 'transcripts/surveys.json')
    surveys = read_json(survey_path)[1]

    results_csv = os.path.join(result_path, 'batch_results.csv')
    code_to_wid = read_results_csv(results_csv)
    chat_workers = chat_to_worker_id(cursor, code_to_wid)

    update_stats(chat_workers, chats, surveys, worker_stats, worker_chats)

def print_chat(chat):
    for event in chat['events']:
        if event['action'] in ('message', 'offer'):
            print event['agent'], event['data']

def normalize_stats(worker_stats):
    keys = ('num_turns', 'num_tokens', 'win_rate', 'negotiator', 'fluent', 'persuasive')
    new_stats = defaultdict(dict)
    worker_stats = worker_stats.items()
    for k in keys:
        stats = normalize([stats[k]['mean'] for worker_id, stats in worker_stats])
        for (worker_id, _), s in izip(worker_stats, stats):
            new_stats[worker_id][k] = s
    return new_stats

def compute_score(stats):
    score = stats['num_turns'] + 2 * stats['num_tokens'] + stats['negotiator'] + stats['fluent'] + stats['persuasive'] + 0.5 * stats['win_rate']
    return score

def score_workers(worker_stats):
    worker_stats = normalize_stats(worker_stats)
    worker_scores = {}
    for worker_id, stats in worker_stats.iteritems():
        worker_scores[worker_id] = compute_score(stats)
    return worker_scores, worker_stats

def assign_qualification(mturk_conn, worker_scores, qual_type, threshold=None, debug=False):
    if not threshold:
        threshold = np.median(worker_scores.values())
    for worker_id, score in worker_scores.iteritems():
        if score > threshold:
            qual = qual_type['good']
        else:
            qual = qual_type['bad']
        # Remove old qual
        if worker_id in worker_quals:
            old_qual = worker_quals[worker_id]
            if old_qual == qual:
                print 'Keep qual {qual_type} of worker {worker_id}'.format(qual_type=old_qual, worker_id=worker_id)
                continue
            else:
                print 'Revoke qual {qual_type} of worker {worker_id}'.format(qual_type=old_qual, worker_id=worker_id)
                if not debug:
                    try:
                        mturk_conn.revoke_qualification(worker_id, old_qual)
                    except MTurkRequestError as e:
                        print "FAILED:", e.reason

        # Only assign bad quals
        if qual == qual_type['bad']:
            print 'Assign {qual_type} to worker {worker_id}'.format(qual_type=qual, worker_id=worker_id)
            if not debug:
                worker_quals[worker_id] = qual
                mturk_conn.assign_qualification(qual, worker_id, send_notification=False)

#def assign_qualification(mturk_conn, worker_scores, qual_type, debug=False):
#    for worker_id, score in worker_scores.iteritems():
#        # Update old qual
#        if worker_id in worker_quals:
#            print 'Update qual {qual_type} of worker {worker_id} from {old_score} to {new_score}'.format(qual_type=qual_type, worker_id=worker_id, old_score=worker_quals[worker_id], new_score=score)
#            if not debug:
#                mturk_conn.update_qualification_score(qual_type, worker_id, score)
#        print 'Assign {qual_type} {score} to worker {worker_id}'.format(qual_type=qual_type, worker_id=worker_id, score=score)
#        if not debug:
#            worker_quals[worker_id] = (qual_type, score)
#            mturk_conn.assign_qualification(qual_type, worker_id, value=score, send_notification=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='data/aws_config.json',
                        help='Config file containing AWS access key and secret access key. '
                             'See data/sample_aws_config.json for an example.')
    parser.add_argument('--debug', action='store_true', help="If provided, runs script in debug mode")
    parser.add_argument('--batches', type=str, nargs='+', required=True, help="Batch names, data is in result_dir/batch_name/*")
    parser.add_argument('--result-dir', type=str, required=True, help='Directory of databases containing chat outcomes.')
    parser.add_argument('--verbose', action='store_true', help='Print chats of good/bad negotiators')
    parser.add_argument('--worker-quals', type=str, help='Path to assigned worker qualifications')


    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))
    debug = args.debug

    host = 'mechanicalturk.sandbox.amazonaws.com'
    if not debug:
        host = 'mechanicalturk.amazonaws.com'

    mturk_connection = MTurkConnection(aws_access_key_id=config["access_key"],
                                       aws_secret_access_key=config["secret_key"],
                                       host=host)

    #mturk_connection.revoke_qualification('AKVDY8OXNMQED', config['quals']['bad'])
    #import sys; sys.exit()

    # Collect worker performance data
    worker_stats = defaultdict(dict)
    worker_chats = defaultdict(list)
    for batch in args.batches:
        result_path = os.path.join(args.result_dir, batch)
        print 'Reading', result_path
        update_worker_stats(result_path, worker_stats, worker_chats)

    def avg_num_chats(worker_ids):
        return np.mean([len(worker_chats[worker_id]) for worker_id in worker_ids])

    worker_scores, worker_stats = score_workers(worker_stats)
    sorted_workers = sorted(worker_scores.items(), key=lambda x: x[1], reverse=True)

    scores = [x[1] for x in sorted_workers]
    mean = np.mean(scores)
    print 'Total #workers:', len(sorted_workers)
    print 'Mean score:', mean, '#Qualified:', len([x for x in scores if x > mean])
    print 'Median score:', np.median(scores), '#Qualified:', len(scores) / 2
    threshold = mean

    if args.verbose:
        for i, (worker_id, score) in enumerate(sorted_workers):
            print '========================='
            print '{} {} score={} #hits={}'.format(i, worker_id, score, len(worker_chats[worker_id]))
            print worker_stats[worker_id]
            for agent_id, chat in worker_chats[worker_id]:
                print '==========chat %s==========' % agent_id
                print_chat(chat)

    if debug:
        print "Running script in debug mode; this won't actually assign any qualifications! " \
              "To confirm these qualifications, run the script with --mode set to PROD"

    worker_quals = read_json(args.worker_quals)

    assign_qualification(mturk_connection, worker_scores, config['quals'], threshold=threshold, debug=debug)

    write_json(worker_quals, args.worker_quals)


