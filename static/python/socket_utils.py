from flask_socketio import SocketIO

socketio = SocketIO()

class LogEmitter:
    def __init__(self, socketio):
        self.socketio = socketio

    def emit_log_message(self, message):
        self.socketio.emit('log_message', {'message': message})