"""Flask + Socket.IO server setup."""

import logging

from flask import Flask
from flask_socketio import SocketIO

from app.config import STATIC_DIR

logger = logging.getLogger(__name__)

socketio = SocketIO(cors_allowed_origins="*", async_mode="eventlet")


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        static_folder=str(STATIC_DIR),
        static_url_path="/static",
    )
    app.config["SECRET_KEY"] = "music-transposer-dev"

    # Register routes
    from app.web.routes import bp as routes_bp
    app.register_blueprint(routes_bp)

    # Register Socket.IO events
    from app.web import events  # noqa: F401 — registers event handlers on import

    # Initialize Socket.IO
    socketio.init_app(app)

    logger.info("Flask app created (static_dir=%s)", STATIC_DIR)
    return app
