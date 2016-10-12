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
    backend.create_user_if_necessary(userid())

    key = request.args.get('key')
    if 'key' in session and session['key'] != key:
        if backend.is_connected(userid()):
            return render_template('error.html')
        else:
            session['key'] = key
    elif 'key' not in session:
        session['key'] = key

    debug = True if request.args.get('debug') is not None and request.args.get('debug') == '1' else False
    status = backend.get_updated_status(userid())
    logger.info("Got updated status %s for user %s" % (Status._names[status], userid()[:6]))
    session["mturk"] = True if request.args.get('mturk') and int(request.args.get('mturk')) == 1 else None
    if session["mturk"]:
        logger.debug("User %s is from Mechanical Turk" % userid()[:6])
    if status == Status.Waiting:
        logger.info("Getting waiting information for user %s" % userid()[:6])
        waiting_info = backend.get_waiting_info(userid())
        return render_template('waiting.html',
                               seconds_until_expiration=waiting_info.num_seconds,
                               waiting_message=waiting_info.message)
    elif status == Status.SingleTask:
        logger.info("Getting single task information for user %s" % userid()[:6])
        single_task_info = backend.get_single_task_info(userid())
        presentation_config = app.config["user_params"]["status_params"]["chat"]["presentation_config"]
        return render_template('single_task.html',
                               scenario=single_task_info.scenario,
                               agent=single_task_info.agent_info,
                               config=presentation_config,
                               num_seconds=single_task_info.num_seconds)
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
        presentation_config = app.config["user_params"]["status_params"]["chat"]["presentation_config"]
        session["room"] = chat_info.room_id
        bot = 0
        if backend.is_user_partner_bot(userid()):
            bot = 1
        return render_template('chat.html',
                               room=chat_info.room_id,
                               scenario=chat_info.scenario,
                               agent=chat_info.agent_info,
                               num_seconds=chat_info.num_seconds,
                               config=presentation_config,
                               bot=bot,
                               debug=debug,
                               partner=chat_info.partner_info)
    elif status == Status.Survey:
        return render_template('survey.html')


def clear_session():
    if "__clear__" in session and session["__clear__"]:
        session["room"] = -1
        session["mturk"] = None
        session["__clear__"] = False
