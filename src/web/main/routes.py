from collections import defaultdict

__author__ = 'anushabala'

from flask import session, render_template, request, redirect, url_for
from flask import current_app as app
from . import main
from web_utils import get_backend
import uuid
from backend import Status
import logging

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.FileHandler("chat.log")
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)


def set_or_get_userid():
    if "sid" in session and session["sid"]:
        return userid()
    session["sid"] = request.cookies.get(app.session_cookie_name)
    if not session["sid"]:
        session["sid"] = str(uuid.uuid4().hex)
    return session["sid"]


def userid():
    return session["sid"]


def generate_unique_key():
    return str(uuid.uuid4().hex)


@main.route('/index', methods=['GET', 'POST'])
@main.route('/', methods=['GET', 'POST'])
def index():
    """Chat room. The user's name and room must be stored in
    the session."""

    set_or_get_userid()
    if not request.args.get('key'):
        return redirect(url_for('main.index', key=generate_unique_key(), **request.args))

    backend = get_backend()
    backend.create_user_if_not_exists(userid())

    key = request.args.get('key')
    if 'key' in session and session['key'] != key:
        if backend.is_connected(userid()):
            return render_template('error.html')
        else:
            session['key'] = key
    elif 'key' not in session:
        session['key'] = key

    status = backend.get_updated_status(userid())

    if not request.args.get('status') or not request.args.get('id'):
        return redirect(url_for('main.index', status=status.lower(), id=userid(), **request.args))

    logger.info("Got updated status %s for user %s" % (status, userid()[:6]))

    session["mturk"] = True if request.args.get('mturk') and int(request.args.get('mturk')) == 1 else None
    if session["mturk"]:
        logger.debug("User %s is from Mechanical Turk" % userid()[:6])
    if status == Status.Waiting:
        logger.info("Getting waiting information for user %s" % userid()[:6])
        waiting_info = backend.get_waiting_info(userid())
        return render_template('waiting.html',
                               seconds_until_expiration=waiting_info.num_seconds,
                               waiting_message=waiting_info.message)
    elif status == Status.Finished:
        logger.info("Getting finished information for user %s" % userid()[:6])
        finished_info = backend.get_finished_info(userid(), from_mturk=session["mturk"])
        session["__clear__"] = True
        mturk_code = finished_info.mturk_code if session["mturk"] else None
        clear_session()
        return render_template('finished.html',
                               finished_message=finished_info.message,
                               mturk_code=mturk_code)
    elif status == Status.Chat:
        logger.info("Getting chat information for user %s" % userid()[:6])
        chat_info = backend.get_chat_info(userid())
        session["room"] = chat_info.room_id
        schema = backend.get_schema()
        if not request.args.get('scenario') or request.args.get('scenario') != chat_info.scenario_id:
            return redirect(url_for('main.index', scenario=chat_info.scenario_id, **request.args))
        return render_template('chat.html',
                               room=chat_info.room_id,
                               kb=chat_info.kb.to_dict(),
                               attributes=[attr.name for attr in schema.attributes],
                               num_seconds=chat_info.num_seconds)
    elif status == Status.Survey:
        return render_template('survey.html')


def clear_session():
    if "__clear__" in session and session["__clear__"]:
        session["room"] = -1
        session["mturk"] = None
        session["__clear__"] = False
