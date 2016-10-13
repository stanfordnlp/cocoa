__author__ = 'anushabala'

from flask import Flask
from flask.ext.socketio import SocketIO
from flask import g
# from multiprocessing import Process, Queue
socketio = SocketIO()
controller_process = None


#@app.teardown_appcontext
def close_connection(exception):
    backend = getattr(g, '_backend', None)
    if backend is not None:
        backend.close()
    # if controller_process is not None:
    #     controller_process.terminate()


def create_app(debug=False, templates_dir='templates'):
    """Create an application."""
    global controller_process

    app = Flask(__name__, template_folder=templates_dir)
    app.debug = debug
    app.config['SECRET_KEY'] = 'gjr39dkjn344_!67#'

    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    # controller_queue = Queue()
    # app.config['controller_queue'] = controller_queue
    # controller_process = Process(target=run_controllers, args=(controller_queue,))
    # controller_process.start()
    app.teardown_appcontext_funcs = [close_connection]

    socketio.init_app(app)
    return app

