__author__ = 'anushabala'

import uuid
from datetime import datetime
import time

from flask import jsonify, render_template, request, redirect, url_for, Markup

from flask import current_app as app

from . import main
from cocoa.web.main.web_utils import get_backend
from cocoa.web.main.backend_utils import Status
from cocoa.basic.event import Event
from cocoa.scripts.html_visualizer import NegotiationHTMLVisualizer
import src.config as task_config


def generate_userid(prefix="U_"):
    return prefix + uuid.uuid4().hex


def userid():
    return request.args.get('uid')


def userid_prefix():
    return userid()[:6]


def generate_unique_key():
    return str(uuid.uuid4().hex)


def format_message(message, status_message):
    timestamp = datetime.now().strftime(u'%x %X')
    left_delim = u"<" if status_message else u""
    right_delim = u">" if status_message else u""
    return u"[{}] {}{}{}".format(timestamp, left_delim, message, right_delim)


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
        return jsonify(valid=True)
    else:
        return jsonify(valid=False, message=backend.get_user_message(userid()))


@main.route('/_submit_survey/', methods=['POST'])
def submit_survey():
    backend = get_backend()
    data = request.json['response']
    uid = request.json['uid']
    backend.submit_survey(uid, data)
    return jsonify(success=True)


@main.route('/_check_inbox/', methods=['GET'])
def check_inbox():
    backend = get_backend()
    uid = userid()
    event = backend.receive(uid)
    if event is not None:
        message = None
        if event.action == 'message':
            message = format_message(u"Partner: {}".format(event.data), False)
        elif event.action == 'join':
            message = format_message("Your partner has joined the room.", True)
        elif event.action == 'leave':
            message = format_message("Your partner has left the room.", True)
        elif event.action == 'select':
            ordered_item = backend.schema.get_ordered_item(event.data)
            message = format_message("Your partner selected: {}".format(", ".join([v[1] for v in ordered_item])),
                                     True)
        elif event.action == 'offer':
            message = format_message("Your partner made an offer. View it on the right and accept or reject it.", True)
            if 'sides' not in event.data.keys():
                sides = None
            return jsonify(message=message, received=True, price=event.data['price'], sides=None, timestamp=event.time)

        elif event.action == 'accept':
            message = format_message("Congrats, your partner accepted your offer!", True)
            return jsonify(message=message, received=True, timestamp=event.time)
        elif event.action == 'reject':
            message = format_message("Sorry, your partner rejected your offer.", True)
            return jsonify(message=message, received=True, timestamp=event.time)
        elif event.action == 'typing':
            if event.data == 'started':
                message = "Your partner is typing..."
            else:
                message = ""
            return jsonify(message=message, status=True, received=True, timestamp=event.time)
        elif event.action == 'eval':
            return jsonify(status=False, received=True, timestamp=event.time)
        return jsonify(message=message, status=False, received=True, timestamp=event.time)
    return jsonify(status=False, received=False)


@main.route('/_typing_event/', methods=['GET'])
def typing_event():
    backend = get_backend()
    action = request.args.get('action')

    uid = userid()
    chat_info = backend.get_chat_info(uid)
    backend.send(uid,
                 Event.TypingEvent(chat_info.agent_index,
                                   action,
                                   str(time.time())))

    return jsonify(success=True)


@main.route('/_send_message/', methods=['GET'])
def text():
    backend = get_backend()
    message = unicode(request.args.get('message'))
    displayed_message = format_message(u"You: {}".format(message), False)
    uid = userid()
    time_taken = float(request.args.get('time_taken'))
    received_time = time.time()
    start_time = received_time - time_taken
    chat_info = backend.get_chat_info(uid)
    backend.send(uid,
                 Event.MessageEvent(chat_info.agent_index,
                                    message,
                                    str(received_time),
                                    str(start_time))
                 )
    return jsonify(message=displayed_message, timestamp=str(received_time))

@main.route('/_send_eval/', methods=['POST'])
def send_eval():
    backend = get_backend()
    labels = request.json['labels']
    eval_data = request.json['eval_data']
    uid = request.json['uid']
    chat_info = backend.get_chat_info(uid)
    data = {'utterance': eval_data['utterance'], 'labels': labels}
    backend.send(uid,
                 Event.EvalEvent(chat_info.agent_index,
                                    data,
                                    eval_data['timestamp'])
                 )
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
                                      str(time.time())))
    return jsonify(message=format_message("You entered the room.", True))


@main.route('/_leave_chat/', methods=['GET'])
def leave_chat():
    backend = get_backend()
    uid = userid()
    chat_info = backend.get_chat_info(uid)
    backend.send(uid, Event.LeaveEvent(chat_info.agent_index,
                                       uid,
                                       str(time.time())))
    return jsonify(success=True)


@main.route('/_check_status_change/', methods=['GET'])
def check_status_change():
    backend = get_backend()
    uid = userid()
    assumed_status = request.args.get('assumed_status')
    if backend.is_status_unchanged(uid, assumed_status):
        return jsonify(status_change=False)
    else:
        return jsonify(status_change=True)


@main.route('/index', methods=['GET', 'POST'])
@main.route('/', methods=['GET', 'POST'])
def index():
    """Chat room. The user's name and room must be stored in
    the session."""

    if not request.args.get('uid'):
        prefix = "U_"
        if request.args.get('mturk') and int(request.args.get('mturk')) == 1:
            # link for Turkers
            prefix = "MT_"
        elif request.args.get('nlp') and int(request.args.get('nlp')) == 1:
            # link for NLP group
            prefix = "NLP_"
        elif request.args.get('bus') and int(request.args.get('bus')) == 1:
            # business school link
            prefix = "BUS_"

        return redirect(url_for('main.index', uid=generate_userid(prefix), **request.args))

    backend = get_backend()

    backend.create_user_if_not_exists(userid())

    status = backend.get_updated_status(userid())

    mturk = True if request.args.get('mturk') and int(request.args.get('mturk')) == 1 else None
    if status == Status.Waiting:
        waiting_info = backend.get_waiting_info(userid())
        return render_template('waiting.html',
                               seconds_until_expiration=waiting_info.num_seconds,
                               waiting_message=waiting_info.message,
                               uid=userid(),
                               title=app.config['task_title'],
                               icon=app.config['task_icon'])
    elif status == Status.Finished:
        finished_info = backend.get_finished_info(userid(), from_mturk=mturk)
        mturk_code = finished_info.mturk_code if mturk else None
        visualize_link = False
        if request.args.get('debug') is not None and request.args.get('debug') == '1':
            visualize_link = True
        return render_template('finished.html',
                               finished_message=finished_info.message,
                               mturk_code=mturk_code,
                               title=app.config['task_title'],
                               icon=app.config['task_icon'],
                               visualize=visualize_link,
                               uid=userid())
    elif status == Status.Incomplete:
        finished_info = backend.get_finished_info(userid(), from_mturk=False, current_status=Status.Incomplete)
        mturk_code = finished_info.mturk_code if mturk else None
        return render_template('finished.html',
                               finished_message=finished_info.message,
                               mturk_code=mturk_code,
                               title=app.config['task_title'],
                               icon=app.config['task_icon'],
                               visualize=False,
                               uid=userid())
    elif status == Status.Chat:
        debug = False
        partner_kb = None
        if request.args.get('debug') is not None and request.args.get('debug') == '1':
            debug = True
        chat_info = backend.get_chat_info(userid())
        return render_template('chat.html',
                               debug=debug,
                               uid=userid(),
                               kb=chat_info.kb.to_dict(),
                               attributes=[attr.name for attr in chat_info.attributes],
                               num_seconds=chat_info.num_seconds,
                               title=app.config['task_title'],
                               instructions=Markup(app.config['instructions']),
                               icon=app.config['task_icon'],
                               partner_kb=partner_kb,
                               quit_enabled=app.config['user_params']['skip_chat_enabled'],
                               quit_after=app.config['user_params']['status_params']['chat']['num_seconds'] -
                                          app.config['user_params']['quit_after'])
    elif status == Status.Survey:
        survey_info = backend.get_survey_info(userid())
        visualization = None
        if task_config.task == task_config.Negotiation:
            complete_chat = backend.get_most_recent_chat(userid())
            agent_idx = backend.get_agent_idx(userid())
            visualization = {
                'chat': complete_chat['events'],
                'agent_idx': agent_idx
            }
        return render_template('task_survey.html',
                               title=app.config['task_title'],
                               uid=userid(),
                               icon=app.config['task_icon'],
                               kb=survey_info.kb.to_dict(),
                               partner_kb=survey_info.partner_kb.to_dict(),
                               attributes=[attr.name for attr in survey_info.attributes],
                               message=survey_info.message,
                               results=survey_info.result,
                               agent_idx=survey_info.agent_idx,
                               scenario_id=survey_info.scenario_id,
                               visualization=visualization)
    elif status == Status.Reporting:
        return render_template('report.html',
                               title=app.config['task_title'],
                               uid=userid(),
                               icon=app.config['task_icon'])


@main.route('/_select_option/', methods=['GET'])
def select():
    backend = get_backend()
    selection_id = int(request.args.get('selection'))
    if selection_id == -1:
        return
    selected_item = backend.select(userid(), selection_id)

    ordered_item = backend.schema.get_ordered_item(selected_item)
    displayed_message = format_message("You selected: {}".format(", ".join([v[1] for v in ordered_item])), True)
    return jsonify(message=displayed_message)


@main.route('/_offer/', methods=['GET'])
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


@main.route('/_accept_offer/', methods=['GET'])
def accept_offer():
    backend = get_backend()
    backend.accept_offer(userid())

    msg = format_message("You accepted the offer!", True)
    return jsonify(message=msg)


@main.route('/_reject_offer/', methods=['GET'])
def reject_offer():
    backend = get_backend()
    backend.reject_offer(userid())

    msg = format_message("You rejected the offer.", True)
    return jsonify(message=msg)


@main.route('/_quit/', methods=['GET'])
def quit():
    backend = get_backend()
    backend.quit(userid())
    displayed_message = format_message("You chose to quit this task.", True)
    return jsonify(message=displayed_message)


@main.route('/_report/', methods=['GET'])
def report():
    backend = get_backend()
    uid = userid()
    feedback = request.args.get('feedback')
    backend.report(uid, feedback)
    return jsonify(success=True)


@main.route('/_init_report/', methods=['GET'])
def init_report():
    backend = get_backend()
    uid = userid()
    backend.init_report(uid)
    return jsonify(success=True)
