import csv
import sqlite3
import json
from argparse import ArgumentParser
from boto.mturk.connection import MTurkConnection
from boto.mturk.price import Price
from boto.mturk.connection import MTurkRequestError

from cocoa.analysis.utils import get_total_tokens_per_agent, get_turns_per_agent, reject_transcript


def process_db(cursor):
    survey_codes = {}
    agent_ids = {}
    cursor.execute('''SELECT * FROM mturk_task''')
    data = cursor.fetchall()
    for uid, code, cid in data:
        if cid not in survey_codes.keys():
            survey_codes[cid] = []
        survey_codes[cid].append((code, uid))

    cursor.execute('''SELECT agent_ids, chat_id FROM chat''')
    data = cursor.fetchall()
    for (u, cid) in data:
        u = json.loads(u)
        user_ids = {0: u["0"], 1: u["0"]}
        agent_ids[cid] = user_ids

    return survey_codes, agent_ids


def is_chat_valid(transcript, idx):
    if "outcome" not in transcript.keys():
        return False

    outcome = transcript["outcome"]

    if outcome is None or outcome["reward"] is None:
        return False

    if outcome["reward"] == 1:
        return True

    return reject_transcript(transcript)


#todo (anusha): copied this from controller.py, needs to be refactored?
def get_winner(transcript):
    scenario = transcript["scenario"]
    kbs = scenario["kbs"]
    offer = transcript["outcome"]["offer"]
    targets = {0: kbs[0]['personal']['Target'],
               1: kbs[1]['personal']['Target']}

    roles = {kbs[0]['personal']['Role']: 0,
             kbs[1]['personal']['Role']: 1}

    diffs = {}

    # for seller, offer is probably lower than target
    seller_idx = roles['seller']
    diffs[seller_idx] = targets[seller_idx] - offer
    if diffs[seller_idx] < 0:
        diffs[seller_idx] = -diffs[seller_idx]

    # for buyer, offer is probably
    buyer_idx = roles['buyer']
    diffs[buyer_idx] = offer - targets[buyer_idx]
    if diffs[buyer_idx] < 0:
        diffs[buyer_idx] = -diffs[buyer_idx]

    if diffs[0] < diffs[1]:
        return 0
    elif diffs[1] < diffs[0]:
        return 1

    return -1


def is_partial_chat(transcript, idx):
    outcome = transcript["outcome"]
    if is_chat_valid(transcript, idx) and outcome["reward"] == 0:
        return reject_transcript(transcript)

    return False


def get_bonus_users(transcript):
    # assume that chat has already been checked to be valid
    return get_winner(transcript)


def read_chats(transcripts_file):
    inp = json.load(open(transcripts_file, 'r'))
    return dict((t["uuid"], t) for t in inp)


def process_hits(survey_codes, agent_ids):
    rejected_codes = {}
    partial_credit = {}
    bonus_codes = {}
    for cid in survey_codes.keys():
        for (code, uid) in survey_codes[cid]:
            idx = 0 if agent_ids[cid][0] == uid else 1
            if not is_chat_valid(all_chats[cid], idx):
                print "REJECT: chat %s, agent %d" % (cid, idx)
                rejected_codes[code] = (cid, idx)
            elif is_partial_chat(all_chats[cid], idx):
                print "PARTIAL CREDIT: chat %s, agent %d " % (cid, idx)
                partial_credit[code] = (cid, idx)
            else:
                winner = get_bonus_users(all_chats[cid])
                if winner >= 0:
                    if winner == idx:
                        # print "BONUS: agent %d in chat %s" % (winner, cid)
                        bonus_codes[code] = (cid, idx)
                    # else:
                    #     print "PARTNER BONUS: agent %d in chat %s" % (1-winner, cid)
                else:
                    print "VALID, NO WINNER: chat %s" % cid

    return rejected_codes, partial_credit, bonus_codes


def make_payments(mturk_conn, results_csv, bonus_amount, rejected, partial, bonuses, debug=False):
    reader = csv.reader(open(results_csv, 'r'))
    header = reader.next()
    assignment_idx = header.index('AssignmentId')
    worker_idx = header.index('WorkerId')
    code_idx = header.index('Answer.surveycode')
    for row in reader:
        assignmentid = row[assignment_idx]
        workerid = row[worker_idx]
        code = row[code_idx]
        try:
            if code in rejected.keys():
                print "Rejecting assignment %s: chat %s, agent %d, worker %s" % (assignmentid, rejected[code][0], rejected[code][1], workerid)
                if not debug:
                    mturk_conn.reject_assignment(assignmentid,
                                                 feedback='Not enough of an attempt to complete the negotiation.')
            elif code in partial.keys():
                print "Partial: approving assignment %s: chat %s, agent %d, worker %s" % (assignmentid, partial[code][0], partial[code][1], workerid)
                if not debug:
                    mturk_conn.approve_assignment(assignmentid,
                                                 feedback='Thanks for attempting this negotiation task! :)')
            else:
                # print "Approve assignment %s: " % code
                if not debug:
                    mturk_conn.approve_assignment(assignmentid,
                                                  feedback='Thanks for attempting this negotiation task! :)')
        except MTurkRequestError as e:
            print "FAILED: approve/reject:", e.reason

        try:
            if code in bonuses.keys():
                print "Bonus for assignment %s: chat %s, agent %d, worker %s" % (assignmentid, bonuses[code][0], bonuses[code][1], workerid)
                if not debug:
                    mturk_conn.grant_bonus(workerid, assignmentid, Price(amount=bonus_amount),
                                           reason='For great negotiation skills!')
            pass
        except MTurkRequestError as e:
            print "FAILED: bonus: ", e.reason

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='data/aws_config.json',
                        help='Config file containing AWS access key and secret access key. '
                             'See data/sample_aws_config.json for an example.')
    parser.add_argument('--debug', action='store_true', help="If provided, runs script in debug mode")
    parser.add_argument('--results', type=str, required=True, help="Path to CSV results from MTurk")
    parser.add_argument('--db', type=str, required=True, help='Path to database containing chat outcomes.')
    parser.add_argument('--transcripts', type=str, required=True, help='Path to transcripts.json file containing chats')
    parser.add_argument('--bonus', type=float, default=0.25,
                        help='Amount to grant as bonus to each worker per assignment.')
    parser.add_argument('--partial', type=float, default=0.15, help='Partial amount for incomplete dialogues')


    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))
    debug = args.debug
    db = args.db
    results_file = args.results
    bonus = args.bonus
    partial_reward = args.partial

    host = 'mechanicalturk.sandbox.amazonaws.com'
    if not debug:
        host = 'mechanicalturk.amazonaws.com'

    mturk_connection = MTurkConnection(aws_access_key_id=config["access_key"],
                                       aws_secret_access_key=config["secret_key"],
                                       host=host)

    all_chats = read_chats(args.transcripts)
    db_connection = sqlite3.connect(db)
    with db_connection:
        cursor = db_connection.cursor()
        survey_codes, agent_ids = process_db(cursor)

    db_connection.close()

    rejected, partial, bonuses = process_hits(survey_codes, agent_ids)
    if debug:
        print "Running script in debug mode; this won't actually accept/reject any HITs! " \
              "To confirm these acceptances/rejections, run the script with --mode set to PROD"
    make_payments(mturk_connection, results_file, bonus, rejected, partial, bonuses, debug)


