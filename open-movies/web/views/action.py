from flask import Blueprint, jsonify, request

from cocoa.web.views.utils import userid, format_message

from web.main.backend import Backend
get_backend = Backend.get_backend

action = Blueprint('action', __name__)

# @action.route('/_select/', methods=['GET'])
# def select():
#     backend = get_backend()
#     book = int(request.args.get('book-split'))
#     hat = int(request.args.get('hat-split'))
#     ball = int(request.args.get('ball-split'))

#     proposal = {'book': book, 'hat': hat, 'ball': ball}
#     backend.select(userid(), proposal)

#     msg = format_message("You selected items and marked deal as agreed!", True)
#     return jsonify(message=msg)

@action.route('/_done/', methods=['GET'])
def done():
    backend = get_backend()
    backend.done(userid())

    msg = format_message("You are done talking.", True)
    return jsonify(message=msg)
