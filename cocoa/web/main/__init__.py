from flask import g
from flask import current_app as app
from web.main.utils import Messages
from web.main.backend import Backend

def get_backend():
    backend = getattr(g, '_backend', None)
    if backend is None:
        g._backend = Backend(app.config["user_params"],
                         app.config["schema"],
                         app.config["scenario_db"],
                         app.config["systems"],
                         app.config["sessions"],
                         app.config["controller_map"],
                         app.config["num_chats_per_scenario"],
                         Messages,
                         active_system=app.config.get('active_system'),
                         active_scenario=app.config.get('active_scenario'),
                         )
        backend = g._backend
    return backend
