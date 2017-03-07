__author__ = 'anushabala'

from src.web.main.web_utils import get_backend
from flask import jsonify, render_template, request, redirect, url_for, Markup
from flask import current_app as app
from routes import generate_userid, userid, userid_prefix, generate_unique_key, format_message
#
# @main.route('/_select_option/', methods=['GET'])
# def select():
#     backend = get_backend()
#     selection_id = int(request.args.get('selection'))
#     if selection_id == -1:
#         return
#     selected_item = backend.select(userid(), selection_id)
#
#     ordered_item = backend.schema.get_ordered_item(selected_item)
#     displayed_message = format_message("You selected: {}".format(", ".join([v[1] for v in ordered_item])), True)
#     return jsonify(message=displayed_message)