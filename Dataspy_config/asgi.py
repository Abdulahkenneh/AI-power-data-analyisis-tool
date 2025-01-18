"""
ASGI config for Dataspy_config project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/asgi/
"""

import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

# Set the default settings module for the 'asgi' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Dataspy_config.settings')

# Define the ASGI application
application = ProtocolTypeRouter({
    "http": get_asgi_application(),  # Handle HTTP requests
    # Uncomment this section if you're using WebSockets
    # "websocket": AuthMiddlewareStack(
    #     URLRouter(
    #         # Add WebSocket routing here
    #     )
    # ),
})
