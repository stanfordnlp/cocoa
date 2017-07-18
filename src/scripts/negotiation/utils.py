__author__ = 'anushabala'
import numpy as np
from src.model.preprocess import tokenize


BUYER = "buyer"
SELLER = "seller"
ROLES = [BUYER, SELLER]
WINNER = "winner"
LOSER = "loser"
OUTCOMES = [WINNER, LOSER]


def get_turns_per_agent(transcript):
    turns = {0: 0, 1: 0}
    for event in transcript["events"]:
        if event["action"] == "message":
            turns[event["agent"]] += 1

    return turns


def get_avg_time_taken(transcript):
    events = transcript["events"]
    start_time = float(events[0]["time"])
    end_time = float(events[-1]["time"])
    return end_time - start_time


def get_avg_tokens_per_agent(transcript):
    tokens = {0: 0., 1: 0.}
    utterances = {0: 0., 1: 0.}
    for event in transcript["events"]:
        if event["action"] == "message":
            msg_tokens = tokenize(event["data"])
            tokens[event["agent"]] += len(msg_tokens)
            utterances[event["agent"]] += 1

    if utterances[0] != 0:
        tokens[0] /= utterances[0]
    if utterances[1] != 0:
        tokens[1] /= utterances[1]

    return tokens


def get_total_tokens_per_agent(transcript):
    tokens = {0: 0., 1: 0.}
    for event in transcript["events"]:
        if event["action"] == "message":
            msg_tokens = tokenize(event["data"])
            tokens[event["agent"]] += len(msg_tokens)

    return tokens


def get_winner(transcript):
    if transcript["outcome"] is None or transcript["outcome"]["reward"] != 1:
        return None
    final_price = transcript["outcome"]["offer"]["price"]
    scenario = transcript["scenario"]
    roles = {
        scenario["kbs"][0]["personal"]["Role"]: 0,
        scenario["kbs"][1]["personal"]["Role"]: 1
    }

    buyer_target = scenario["kbs"][roles[BUYER]]["personal"]["Target"]
    seller_target = scenario["kbs"][roles[SELLER]]["personal"]["Target"]
    # print "Buyer target: {:.1f}\tList price: {:.1f}\tFinal price: {:.1f}".format(buyer_target, seller_target,
    #                                                                              final_price)
    if np.abs(buyer_target - final_price) < np.abs(seller_target - final_price):
        # print "Buyer won\n"
        return roles[BUYER]
    elif np.abs(seller_target - final_price) < np.abs(buyer_target - final_price):
        # print "Seller won\n"
        return roles[SELLER]
    else:
        # print "Tie"
        return -1


def get_win_margin(transcript):
    winner = get_winner(transcript)
    if winner is None:
        return -1

    scenario = transcript["scenario"]
    final_price = transcript["outcome"]["offer"]["price"]
    if winner == -1:
        target = scenario["kbs"][0]["personal"]["Target"]
        partner_target = scenario["kbs"][1]["personal"]["Target"]
    else:
        target = scenario["kbs"][winner]["personal"]["Target"]
        partner_target = scenario["kbs"][1 - winner]["personal"]["Target"]

    midpoint = np.abs(partner_target - target) / 2.

    margin = np.abs(target - final_price) / midpoint
    if margin > 5:
        print "Winner target: {:.1f}\tPartner target: {:.1f}\tMidpoint: {:.1f}" \
              "\tFinal price: {:.1f}\tMargin: {:.1f}".format(target, partner_target,
                                                             midpoint, final_price,
                                                             margin)
    return margin
