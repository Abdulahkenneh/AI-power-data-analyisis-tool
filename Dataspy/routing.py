# routing.py
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/chat/', consumers.ChatConsumer.as_asgi()),  # Chat endpoint
    re_path(r'ws/notifications/', consumers.NotificationConsumer.as_asgi()),  # New Notification endpoint
]
