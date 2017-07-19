from collections import defaultdict
from boto.mturk.connection import MTurkConnection
from boto.mturk.price import Price
from boto.mturk.connection import MTurkRequestError

__author__ = 'anushabala'
from argparse import ArgumentParser
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import json

DATE_FMT = "%a %b %d %H:%M:%S %Z %Y"


def assign_bonus(duration):
    if 7 < duration <= 10:
        return 0.25
    elif 10 < duration <= 13:
        return 0.5
    elif 13 < duration <= 25:
        return 0.75
    elif duration > 25:
        print "t>25"
        return 2
    else:
        return 0.


def grant_bonuses():
    reader = csv.reader(results_file)
    header = reader.next()
    response_col = header.index('Answer.surveycode')
    feedback_col = header.index('Answer.feedback')
    hit_col = header.index('AssignmentId')
    worker_col = header.index('WorkerId')
    total_bonus = 0.

    for row in reader:
        response = row[response_col]
        hit_id = row[hit_col]
        worker_id = row[worker_col]
        feedback = row[feedback_col]
        if response.startswith("I") or response.startswith("C"):
            response = "MTURK_TASK_" + response
        if response == "{}":
            print "Empty response; feedback: {:s}".format(feedback)
            if feedback != "{}" and "MTURK_TASK_" not in feedback:
                total_bonus += 0.25
                print "Based on feedback: Granting bonus 0.25 to worker {:s}".format(worker_id)
                if not debug:
                    mturk_connection.grant_bonus(worker_id, hit_id, Price(amount=0.25),
                                                 reason='For the trouble you experienced with our negotiation platform')
            elif "MTURK_TASK_" in feedback:
                print "Survey code was entered in feedback? {:s} Trying..".format(feedback)
                response = feedback
        if response not in mturk_codes.keys():
            print "No userid associated with code {:s}".format(response)
            continue
        uid = mturk_codes[response]
        if uid not in bonuses.keys():
            print "No chats for user {:s} with code {:s}".format(uid, response)
            continue

        bonus = bonuses[uid]
        if bonus > 0.:
            total_bonus += bonus
            print "Granting bonus {:.2f} to worker {:s}".format(bonus, worker_id)
            if not debug:
                mturk_connection.grant_bonus(worker_id, hit_id, Price(amount=bonus),
                                             reason='For having to spend extra time on our negotiation platform')

    print "Total bonus awarded: {:.2f}".format(total_bonus)


def get_last_event(cid):
    cursor.execute('''SELECT time FROM event WHERE chat_id=?''', (cid,))
    times = cursor.fetchall()
    if times is None:
        return None
    max_time = -1
    for event_time in times:
        event_time = float(event_time[0])
        max_time = event_time if event_time > max_time else max_time
    return max_time


def get_user_chats():
    user_chats = defaultdict(list)
    chat_times = defaultdict(float)
    cursor.execute('''SELECT chat_id, agent_ids, start_time FROM chat''')
    for (cid, agent_ids, start_time) in cursor.fetchall():
        agent_ids = json.loads(agent_ids)
        start_time = float(start_time)
        user_chats[agent_ids["0"]].append(cid)
        user_chats[agent_ids["1"]].append(cid)
        end_time = get_last_event(cid)
        chat_times[cid] = (end_time - start_time)

    return user_chats, chat_times


def get_mturk_codes():
    codes = {}
    cursor.execute('''SELECT * FROM mturk_task WHERE mturk_code IS NOT NULL''')
    for (uid, c, _) in cursor.fetchall():
        codes[c] = uid
    return codes


def get_bonuses():
    times = []
    num_chats = []
    bonuses = {}
    for uid in user_chats.keys():
        t = sum([chat_times[cid] for cid in user_chats[uid]]) / 60.
        if t < 1:
            continue
        num_chats.append(len(user_chats[uid]))
        times.append(t)
        bonuses[uid] = assign_bonus(t)

    print "Avg time taken: {:.2f}".format(np.mean(times))
    print "Std: {:.2f}".format(np.std(times))
    print "Average number of chats: {:.2f}".format(np.mean(num_chats))
    print "Std: {:.2f}".format(np.std(num_chats))
    print "Total bonus: ${:.2f}".format(np.sum(bonuses.values()))
    hist, bins = np.histogram(times)
    print bins

    # width = np.diff(bins)
    # center = (bins[:-1] + bins[1:]) / 2
    #
    # fig, ax = plt.subplots(figsize=(8,3))
    # ax.bar(center, hist, align='center', width=width)
    # ax.set_xticks(bins)
    # plt.show()

    return bonuses

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='data/aws_config.json',
                        help='Config file containing AWS access key and secret access key. '
                             'See data/sample_aws_config.json for an example.')
    parser.add_argument('--debug', action='store_true', help="If provided, runs script in debug mode")
    parser.add_argument('--results', type=str, required=True, help='Path to results CSV file from Mechanical Turk')
    parser.add_argument('--db', type=str, required=True, help='Path to website database file')
    parser.add_argument('--min-time', type=float, default=10.0, help='Minimum time to award bonus for')
    args = parser.parse_args()

    config = json.load(open(args.config, 'r'))
    debug = args.debug

    results_file = open(args.results, 'r')
    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()
    user_chats, chat_times = get_user_chats()
    mturk_codes = get_mturk_codes()
    bonuses = get_bonuses()

    if debug:
        print "Running script in debug mode."
        host = 'mechanicalturk.sandbox.amazonaws.com'
    else:
        host = 'mechanicalturk.amazonaws.com'

    mturk_connection = MTurkConnection(aws_access_key_id=config["access_key"],
                                       aws_secret_access_key=config["secret_key"],
                                       host=host)

    grant_bonuses()
