from flask import Blueprint, jsonify, request
from cocoa.web.views.utils import userid, format_message
from web.main.backend import get_backend

action = Blueprint('action', __name__)

@action.route('/_offer/', methods=['GET'])
def offer():
    backend = get_backend()
    price = float(request.args.get('price'))
    sides = request.args.get('sides')

    offer = {'price': price,
             'sides': sides}

    if offer is None or price == -1:
        return jsonify(message=format_message("You made an invalid offer. Please try again.", True))
    backend.make_offer(userid(), offer)

    displayed_message = format_message("You made an offer!", True)
    return jsonify(message=displayed_message)


@action.route('/_select/', methods=['GET'])
def select():
    backend = get_backend()
    book = int(request.args.get('book-split'))
    hat = int(request.args.get('hat-split'))
    ball = int(request.args.get('ball-split'))

    proposal = {'book': book, 'hat': hat, 'ball': ball}
    backend.select(userid(), proposal)

    msg = format_message("You selected items and marked deal as agreed!", True)
    return jsonify(message=msg)

@action.route('/_reject/', methods=['GET'])
def reject():
    backend = get_backend()
    backend.reject(userid())

    msg = format_message("You declared there was no deal!", True)
    return jsonify(message=msg)


@action.route('/_quit/', methods=['GET'])
def quit():
    backend = get_backend()
    backend.quit(userid())
    displayed_message = format_message("You chose to quit this task.", True)
    return jsonify(message=displayed_message)
