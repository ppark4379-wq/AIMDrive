"""Compatibility fix for JSON carried in binary WebSocket frames on ROS1.

Autobahn rosbridge passes binary frames through as bytes. The installed JSON
Protocol.incoming then applies str(bytes), adding b'...' around valid JSON.
Route JSON-mode binary frames through the existing UTF-8 text decoding branch.
BSON mode, authentication, ordering, payloads and queue policy remain upstream's.
"""
from functools import wraps


def install_binary_json_fix(websocket_class):
    original = websocket_class.onMessage
    if getattr(original, '_stage6e_binary_json_fix', False):
        return

    @wraps(original)
    def on_message(self, message, binary):
        if binary and not self.__class__.bson_only_mode:
            binary = False
        return original(self, message, binary)

    on_message._stage6e_binary_json_fix = True
    websocket_class.onMessage = on_message
