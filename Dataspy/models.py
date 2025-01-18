from django.db import models
from django.conf import settings
from django.core.exceptions import ValidationError

class UserProfile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    url = models.URLField(blank=True, null=True)
    date_created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.user.username

class DataQuery(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    prompt = models.TextField()
    response = models.TextField()
    file_path = models.CharField(max_length=255)
    columns = models.JSONField()
    data_types = models.JSONField()
    head = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Data Query by {self.user} on {self.created_at}"

class ChatMessage(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)
    user_request = models.TextField(null=True, blank=True)
    code = models.TextField(null=True, blank=True)
    ai_response = models.TextField(null=True, blank=True)
    plotly = models.TextField(null=True, blank=True)  # Store Plotly HTML
    matplot = models.TextField(null=True, blank=True)  # Store base64-encoded Matplotlib plot
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)

    def __str__(self):
        return f"{self.user_request[:20]}: {self.ai_response[:50]}"

# class ChatMessage(models.Model):
#     user_choices = [('user', 'User'), ('chatgpt', 'ChatGPT')]
#     user = models.CharField(max_length=7, choices=user_choices)
#     text = models.TextField()
#     code = models.TextField(null=True, blank=True)
#     ai_response = models.TextField(null=True, blank=True)
#     message_type = models.CharField(max_length=8, choices=[('response', 'Response'), ('prompt', 'Prompt')])
#     graph = models.ImageField(upload_to='graphs/', null=True, blank=True)
#     timestamp = models.DateTimeField(auto_now_add=True)
#
#     def __str__(self):
#         return f"{self.user}: {self.text[:30]}"
#
#


class ErrorLog(models.Model):
    chat_message = models.ForeignKey(ChatMessage, on_delete=models.CASCADE, related_name='errors')
    error_message = models.TextField(null=True,blank=True)
    error_type = models.CharField(max_length=100, null=True, blank=True)
    stack_trace = models.TextField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        chat_message_str = str(self.chat_message) if self.chat_message else "Unknown Chat Message"
        error_message_snippet = self.error_message[:30] if self.error_message else "No error message"
        return f"Error for {chat_message_str}: {error_message_snippet}"

from django.db import models, transaction
from django.core.exceptions import ValidationError
from django.conf import settings

class UserFile(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    uploaded_file = models.FileField(upload_to='uploads/')
    date_uploaded = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                name='limit_user_files',
                fields=['user'],  # Apply constraint to the user field
                condition=models.Q(),  # Placeholder if needed
                deferrable=models.Deferrable.DEFERRED)
        ]







class Video(models.Model):
    title = models.CharField(max_length=100)
    video_file = models.FileField(upload_to='videos/')

    def __str__(self):
        return self.title

class UserloadedFile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    file_path = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Data Query by {self.user} on {self.created_at}"


class Ai_generated_code(models.Model):
    chat_message = models.ForeignKey(ChatMessage,blank=True, null=True,on_delete=models.CASCADE)
    code        =  models.TextField(blank=True,null=True)
    ai_response        =  models.TextField(blank=True,null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Data Query by {self.chat_message} on {self.created_at}"

