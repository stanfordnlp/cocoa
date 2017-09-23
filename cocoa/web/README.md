### Main classes/modules
`cocoa.web` provides basic backend functions follows the structure of a Flask application.
- **Backend** (`main/backend.py`): Manage the database that records user information and the chat log.
- **Routing** (`views/`): Handle requests, render templates, and interact with the backend.

To build you own chat interface, add HTML templates (based on [Jinja2](http://jinja.pocoo.org/docs/2.9/)) in `task/templates`.
