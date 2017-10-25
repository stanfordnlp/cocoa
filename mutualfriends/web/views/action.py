from flask import Blueprint, jsonify, request
from cocoa.web.views.utils import userid, format_message
from web.main.backend import get_backend

action = Blueprint('action', __name__)

@action.route('/_select_option/', methods=['GET'])
def select():
    backend = get_backend()
    selection_id = int(request.args.get('selection'))
    if selection_id == -1:
        return
    selected_item = backend.select(userid(), selection_id)

    ordered_item = backend.schema.get_ordered_item(selected_item)
    displayed_message = format_message("You selected: {}".format(", ".join([v[1] for v in ordered_item])), True)
    return jsonify(message=displayed_message)
