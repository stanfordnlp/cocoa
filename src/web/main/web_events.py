from flask import session, request
from flask.ext.socketio import emit, join_room, leave_room, send
from src.web import socketio
from datetime import datetime
from web_utils import get_backend
from backend import Status
from routes import userid
from src.basic.event import Event
import logging

date_fmt = '%m-%d-%Y:%H-%M-%S'
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.FileHandler("chat.log")
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)


def userid_prefix():
    return userid()[:6]


@socketio.on('connect', namespace='/main')
def connect():
    backend = get_backend()
    backend.connect(userid())
    logger.info("User %s established connection on non-chat template" % userid_prefix())


@socketio.on('connect', namespace='/chat')
def connect():
    backend = get_backend()
    backend.connect(userid())
    logger.info("User %s established connection on chat template" % userid_prefix())


@socketio.on('is_chat_valid', namespace='/chat')
def check_valid_chat(data):
    backend = get_backend()
    uid = userid()
    if backend.is_chat_valid(uid):
        logger.debug("Chat is still valid for user %s" % userid_prefix())
        return {'valid': True}
    else:
        logger.info("Chat is not valid for user %s" % userid_prefix())
        return {'valid': False, 'message': backend.get_user_message(uid)}


@socketio.on('check_status_change', namespace='/main')
def check_status_change(data):
    backend = get_backend()
    assumed_status = Status.from_str(data['current_status'])

    if backend.is_status_unchanged(userid(), assumed_status):
        logger.debug("User %s status unchanged. Status: %s" % (userid_prefix(), Status._names[assumed_status]))
        return {'status_change': False}
    else:
        logger.info("User %s status changed from %s" % (userid_prefix(), Status._names[assumed_status]))
        return {'status_change': True}


@socketio.on('submit_survey', namespace='/main')
def submit_survey(data):
    backend = get_backend()
    logger.debug("User %s submitted survey. Form data: %s" % (userid_prefix(), str(data)))
    backend.submit_survey(userid(), data)


@socketio.on('joined', namespace='/chat')
def joined(message):
    """Sent by clients when they enter a room.
    A status message is broadcast to all people in the room."""
    start_chat()
    join_room(session["room"])
    backend = get_backend()
    logger.debug("User %s joined chat room %d" % (userid_prefix(), session["room"]))

    uid = userid()
    chat_info = backend.get_chat_info(uid)
    backend.send(userid(), Event.JoinEvent(chat_info.agent_index,
                                           uid,
                                           datetime.now().strftime(date_fmt)))
    emit_message_to_self("You entered the room.", status_message=True)


@socketio.on('check_inbox', namespace='/chat')
def check_inbox(data):
    backend = get_backend()
    uid = userid()
    event = backend.receive(uid)
    while event is not None:
        print "Call to check_inbox: received event: {}".format(event)
        if event.action == 'message':
            emit_message_to_self("Friend: {}".format(event.data))
        elif event.action == 'join':
            emit_message_to_self("Your friend has joined the room.")
        elif event.action == 'leave':
            emit_message_to_self("Your friend has left the room.")
        elif event.action == 'select':
            emit_message_to_self("Your friend selected {}".format(", ".join(event.data.values())))
            if backend.is_game_over(uid):
                emit('endchat',
                {'message': "You've completed this task! Redirecting you..."},
                room=session["room"])

        event = backend.receive(uid)


@socketio.on('text', namespace='/chat')
def text(message):
    """Sent by a client when the user entered a new message.
    The message is sent to all people in the room."""
    backend = get_backend()
    msg = message['msg']
    logger.debug("User %s said: %s" % (userid_prefix(), msg))
    emit_message_to_self("You: {}".format(msg))
    uid = userid()
    chat_info = backend.get_chat_info(uid)
    backend.send(uid,
                 Event.MessageEvent(chat_info.agent_index,
                                    msg,
                                    datetime.now().strftime(date_fmt))
                 )


@socketio.on('select', namespace='/chat')
def select(message):
    """Sent by a client when the user entered a new message.
    The message is sent to all people in the room."""
    backend = get_backend()
    selection_id = int(message['selection'])
    if selection_id == -1:
        return
    selected_item = backend.select(userid(), selection_id)

    emit_message_to_self("You selected: {}".format(", ".join(selected_item.values()), status_message=True))


@socketio.on('disconnect', namespace='/chat')
def disconnect():
    """Sent by clients when they leave a room.
    A status message is broadcast to all people in the room."""
    room = session["room"]

    leave_room(room)
    backend = get_backend()
    backend.disconnect(userid())
    logger.info("User %s disconnected from chat and left room %d" % (userid_prefix(), room))


@socketio.on('disconnect', namespace='/main')
def disconnect():
    """
    Called when user disconnects from any state
    :return: No return value
    """
    backend = get_backend()
    backend.disconnect(userid())
    logger.info("User %s disconnected" % (userid_prefix()))


def emit_message_to_self(message, status_message=False):
    timestamp = datetime.now().strftime('%x %X')
    left_delim = "<" if status_message else ""
    right_delim = ">" if status_message else ""
    emit('message', {'msg': "[{}] {}{}{}".format(timestamp, left_delim, message, right_delim)}, room=request.sid)


def emit_message_to_chat_room(message, status_message=False):
    timestamp = datetime.now().strftime('%x %X')
    left_delim = "<" if status_message else ""
    right_delim = ">" if status_message else ""
    emit('message', {'msg': "[{}] {}{}{}".format(timestamp, left_delim, message, right_delim)}, room=session["room"])


# todo we probably don't need this anymore
def emit_message_to_partner(message, status_message=False):
    timestamp = datetime.now().strftime('%x %X')
    left_delim = "<" if status_message else ""
    right_delim = ">" if status_message else ""
    emit('message', {'msg': "[{}] {}{}{}".format(timestamp, left_delim, message, right_delim)}, room=session["room"],
         include_self=False)


def start_chat():
    backend = get_backend()
    uid = userid()
    chat_info = backend.get_chat_info(uid)
    backend.send(uid,
                 Event.JoinEvent(chat_info.agent_index,
                                 uid,
                                 datetime.now().strftime(date_fmt)))

