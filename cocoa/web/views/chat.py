import time
from flask import Blueprint, jsonify, render_template, request, redirect, url_for, Markup
from flask import current_app as app

from utils import generate_userid, userid, format_message
from cocoa.web.main.utils import Status
from cocoa.core.event import Event

from web.main.backend import Backend
get_backend = Backend.get_backend

chat = Blueprint('chat', __name__)

@chat.route('/_connect/', methods=['GET'])
def connect():
    backend = get_backend()
    backend.connect(userid())
    return jsonify(success=True)


@chat.route('/_disconnect/', methods=['GET'])
def disconnect():
    backend = get_backend()
    backend.disconnect(userid())
    return jsonify(success=True)


@chat.route('/_check_chat_valid/', methods=['GET'])
def is_chat_valid():
    backend = get_backend()
    if backend.is_chat_valid(userid()):
        return jsonify(valid=True)
    else:
        return jsonify(valid=False, message=backend.get_user_message(userid()))

@chat.route('/_submit_survey/', methods=['POST'])
def submit_survey():
    backend = get_backend()
    data = request.json['response']
    uid = request.json['uid']
    backend.submit_survey(uid, data)
    return jsonify(success=True)

@chat.route('/_check_inbox/', methods=['GET'])
def check_inbox():
    backend = get_backend()
    uid = userid()
    event = backend.receive(uid)
    if event is not None:
        data = backend.display_received_event(event)
        return jsonify(received=True, timestamp=event.time, **data)
    else:
        return jsonify(received=False)


@chat.route('/_typing_event/', methods=['GET'])
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


@chat.route('/_send_message/', methods=['GET'])
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

@chat.route('/_send_eval/', methods=['POST'])
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

@chat.route('/_join_chat/', methods=['GET'])
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


@chat.route('/_leave_chat/', methods=['GET'])
def leave_chat():
    backend = get_backend()
    uid = userid()
    chat_info = backend.get_chat_info(uid)
    backend.send(uid, Event.LeaveEvent(chat_info.agent_index,
                                       uid,
                                       str(time.time())))
    return jsonify(success=True)


@chat.route('/_check_status_change/', methods=['GET'])
def check_status_change():
    backend = get_backend()
    uid = userid()
    assumed_status = request.args.get('assumed_status')
    if backend.is_status_unchanged(uid, assumed_status):
        return jsonify(status_change=False)
    else:
        return jsonify(status_change=True)

@chat.route('/index', methods=['GET', 'POST'])
@chat.route('/', methods=['GET', 'POST'])
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

        return redirect(url_for('chat.index', uid=generate_userid(prefix), **request.args))

    #if request.args.get('bot'):
    #    app.config['active_system'] = request.args.get('bot')
    #else:
    #    app.config['active_system'] = None

    #if request.args.get('s'):
    #    app.config['active_scenario'] = int(request.args.get('s'))
    #else:
    #    app.config['active_scenario'] = None

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
        #if task_config.task == task_config.Negotiation:
        #    complete_chat = backend.get_most_recent_chat(userid())
        #    agent_idx = backend.get_agent_idx(userid())
        #    visualization = {
        #        'chat': complete_chat['events'],
        #        'agent_idx': agent_idx
        #    }
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

@chat.route('/_report/', methods=['GET'])
def report():
    backend = get_backend()
    uid = userid()
    feedback = request.args.get('feedback')
    backend.report(uid, feedback)
    return jsonify(success=True)


@chat.route('/_init_report/', methods=['GET'])
def init_report():
    backend = get_backend()
    uid = userid()
    backend.init_report(uid)
    return jsonify(success=True)
