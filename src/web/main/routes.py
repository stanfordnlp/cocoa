import json

__author__ = 'anushabala'

import uuid
import logging
from datetime import datetime

from flask import jsonify, render_template, request, redirect, url_for, Markup
from flask import current_app as app

from . import main
from web_utils import get_backend
from backend import Status
from src.basic.event import Event

date_fmt = '%m-%d-%Y:%H-%M-%S'
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.FileHandler("chat.log")
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)


def generate_userid():
    return "U_"+uuid.uuid4().hex


def userid():
    return request.args.get('uid')


def userid_prefix():
    return userid()[:6]


def generate_unique_key():
    return str(uuid.uuid4().hex)


def get_formatted_date():
    return datetime.now().strftime(date_fmt)


# Required args: uid (the user ID of the current user)
@main.route('/_connect/', methods=['GET'])
def connect():
    backend = get_backend()
    backend.connect(userid())
    return jsonify(success=True)


# Required args: uid (the user ID of the current user)
@main.route('/_disconnect/', methods=['GET'])
def disconnect():
    backend = get_backend()
    backend.disconnect(userid())
    return jsonify(success=True)


# Required args: uid (the user ID of the current user)
@main.route('/_check_chat_valid/', methods=['GET'])
def is_chat_valid():
    backend = get_backend()
    if backend.is_chat_valid(userid()):
        logger.debug("Chat is still valid for user %s" % userid_prefix())
        return jsonify(valid=True)
    else:
        logger.info("Chat is not valid for user %s" % userid_prefix())
        return jsonify(valid=False, message=backend.get_user_message(userid()))


@main.route('/_submit_survey/', methods=['POST'])
def submit_survey():
    backend = get_backend()
    data = request.json['response']
    uid = request.json['uid']
    backend.submit_survey(uid, data)
    return jsonify(success=True)


@main.route('/_join_chat/', methods=['GET'])
def join_chat():
    """Sent by clients when they enter a room.
    A status message is broadcast to all people in the room."""
    backend = get_backend()
    uid = userid()
    chat_info = backend.get_chat_info(uid)
    backend.send(uid, Event.JoinEvent(chat_info.agent_index,
                                      uid,
                                      get_formatted_date()))
    return jsonify(message=format_message("You entered the room.", True))


@main.route('/_leave_chat/', methods=['GET'])
def leave_chat():
    backend = get_backend()
    uid = userid()
    chat_info = backend.get_chat_info(uid)
    backend.send(uid, Event.LeaveEvent(chat_info.agent_index,
                                       uid,
                                       get_formatted_date()))
    return jsonify(success=True)


@main.route('/_check_status_change/', methods=['GET'])
def check_status_change():
    backend = get_backend()
    uid = userid()
    assumed_status = request.args.get('assumed_status')
    if backend.is_status_unchanged(uid, assumed_status):
        logger.debug("User %s status unchanged. Status: %s" % (uid, assumed_status))
        return jsonify(status_change=False)
    else:
        logger.info("User %s status changed from %s" % (userid_prefix(), assumed_status))
        return jsonify(status_change=True)


def format_message(message, status_message):
    timestamp = datetime.now().strftime('%x %X')
    left_delim = "<" if status_message else ""
    right_delim = ">" if status_message else ""
    return "[{}] {}{}{}".format(timestamp, left_delim, message, right_delim)


@main.route('/_check_inbox/', methods=['GET'])
def check_inbox():
    backend = get_backend()
    uid = userid()
    event = backend.receive(uid)
    if event is not None:
        message = None
        if event.action == 'message':
            message = format_message("Friend: {}".format(event.data), False)
        elif event.action == 'join':
            message = format_message("Your friend has joined the room.", True)
        elif event.action == 'leave':
            message = format_message("Your friend has left the room.", True)
        elif event.action == 'select':
            message = format_message("Your friend selected {}".format(", ".join([v[1] for v in event.data])), False)
        return jsonify(message=message, received=True)
    return jsonify(received=False)


@main.route('/_send_message/', methods=['GET'])
def text():
    backend = get_backend()
    message = request.args.get('message')
    logger.debug("User %s said: %s" % (userid_prefix(), message))
    displayed_message = format_message("You: {}".format(message), False)
    uid = userid()
    chat_info = backend.get_chat_info(uid)
    backend.send(uid,
                 Event.MessageEvent(chat_info.agent_index,
                                    message,
                                    get_formatted_date())
                 )
    return jsonify(message=displayed_message)


@main.route('/_select_option/', methods=['GET'])
def select():
    backend = get_backend()
    selection_id = int(request.args.get('selection'))
    if selection_id == -1:
        return
    selected_item = backend.select(userid(), selection_id)

    displayed_message = format_message("You selected: {}".format(", ".join([v[1] for v in selected_item])), True)
    return jsonify(message=displayed_message)


@main.route('/index', methods=['GET', 'POST'])
@main.route('/', methods=['GET', 'POST'])
def index():
    """Chat room. The user's name and room must be stored in
    the session."""

    if not request.args.get('uid'):
        return redirect(url_for('main.index', uid=generate_userid(), **request.args))

    backend = get_backend()
    backend.create_user_if_not_exists(userid())

    status = backend.get_updated_status(userid())

    logger.info("Got updated status %s for user %s" % (status, userid()[:6]))

    mturk = True if request.args.get('mturk') and int(request.args.get('mturk')) == 1 else None
    if status == Status.Waiting:
        logger.info("Getting waiting information for user %s" % userid()[:6])
        waiting_info = backend.get_waiting_info(userid())
        return render_template('waiting.html',
                               seconds_until_expiration=waiting_info.num_seconds,
                               waiting_message=waiting_info.message,
                               uid=userid(),
                               title=app.config['task_title'])
    elif status == Status.Finished:
        logger.info("Getting finished information for user %s" % userid()[:6])
        finished_info = backend.get_finished_info(userid(), from_mturk=mturk)
        mturk_code = finished_info.mturk_code if mturk else None
        return render_template('finished.html',
                               finished_message=finished_info.message,
                               mturk_code=mturk_code,
                               title=app.config['task_title'])
    elif status == Status.Chat:
        logger.info("Getting chat information for user %s" % userid()[:6])
        chat_info = backend.get_chat_info(userid())
        schema = backend.get_schema()
        return render_template('chat.html',
                               uid=userid(),
                               kb=chat_info.kb.to_dict(),
                               attributes=[attr.name for attr in schema.attributes],
                               num_seconds=chat_info.num_seconds,
                               title=app.config['task_title'],
                               instructions=Markup(app.config['instructions']))
    elif status == Status.Survey:
        return render_template('survey.html',
                               title=app.config['task_title'],
                               uid=userid())
