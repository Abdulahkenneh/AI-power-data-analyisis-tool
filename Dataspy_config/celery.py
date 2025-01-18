from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Dataspy_config.settings')

app = Celery('Dataspy_config')

# Configure Celery using Django settings with the `CELERY_` namespace.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Explicitly enable broker connection retry on startup (Celery 6.0+ compatible).
app.conf.broker_connection_retry_on_startup = True

# Load task modules from all registered Django app configs.
app.autodiscover_tasks()

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
