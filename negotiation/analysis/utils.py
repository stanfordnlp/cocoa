import numpy as np

from cocoa.core.dataset import Example

from core.scenario import Scenario
from model.preprocess import Preprocessor


BUYER = "buyer"
SELLER = "seller"
ROLES = [BUYER, SELLER]
WINNER = "winner"
LOSER = "loser"
OUTCOMES = [WINNER, LOSER]


def filter_rejected_chats(transcripts):
    filtered = []
    for chat in transcripts:
        ex = Example.from_dict(None, chat, Scenario)
        if not Preprocessor.skip_example(ex):
            filtered.append(chat)
    return filtered


def filter_incomplete_chats(transcripts):
    filtered = []
    for chat in transcripts:
        winner = get_winner(chat)
        if winner is not None:
            filtered.append(chat)
    return filtered


def get_category(transcript):
    return transcript['scenario']['category']




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

def get_margin(transcript, agent=None, role=None):
    if role is not None:
        scenario = transcript["scenario"]
        roles = {
            scenario["kbs"][0]["personal"]["Role"]: 0,
            scenario["kbs"][1]["personal"]["Role"]: 1
        }
        agent = roles[role]

    if agent is None:
        # by default, get margin for winner
        winner = get_winner(transcript)
        if winner is None:
            return -1
        if winner == -1:
            winner = 0
        agent = winner
    scenario = transcript["scenario"]
    final_price = transcript["outcome"]["offer"]["price"]

    agent_role = scenario['kbs'][agent]['personal']['Role']
    agent_target = scenario["kbs"][agent]["personal"]["Target"]
    partner_target = scenario["kbs"][1 - agent]["personal"]["Target"]
    midpoint = (agent_target + partner_target) / 2.
    norm_factor = np.abs(midpoint - agent_target)
    if agent_role == SELLER:
        margin = (final_price - midpoint) / norm_factor
    else:
        margin = (midpoint - final_price) / norm_factor

    # print 'Chat {:s}\tAgent: {:d}\tAgent role: {:s}\tAgent target: {:.2f}\t' \
    #       'Partner target: {:.2f}\tMidpoint: {:.2f}\tNorm factor: {:.2f}\t' \
    #       'Final offer: {:.2f}\tMargin{:.2f}'.format(transcript['uuid'], agent, agent_role,
    #                                                  agent_target, partner_target, midpoint, norm_factor,
    #                                                  final_price, margin)
    return margin
