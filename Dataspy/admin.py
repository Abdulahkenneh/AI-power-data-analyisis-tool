from django.contrib import admin
from .models import UserProfile,Ai_generated_code , UserloadedFile, UserFile, Video, ErrorLog, DataQuery, ChatMessage

class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'url')
    search_fields = ('user__username', 'url')
    ordering = ('user',)

admin.site.register(UserProfile, UserProfileAdmin)

class DataQueryAdmin(admin.ModelAdmin):
    list_filter = ('user',)

admin.site.register(DataQuery, DataQueryAdmin)

class ChatAdmin(admin.ModelAdmin):
    list_filter = ('user',)

admin.site.register(ChatMessage, ChatAdmin)

class UserFileAdmin(admin.ModelAdmin):
    list_display = ( 'uploaded_file', 'date_uploaded')

admin.site.register(UserFile, UserFileAdmin)

class VideoAdmin(admin.ModelAdmin):
    list_display = ('video_file', 'title')

admin.site.register(Video, VideoAdmin)

class ErrorLogAdmin(admin.ModelAdmin):
    list_display = ('chat_message_safe', 'error_message_safe', 'error_type', 'timestamp')
    search_fields = ('error_message', 'error_type')
    list_filter = ('error_type', 'timestamp')


    def chat_message_safe(self, obj):
        return str(obj.chat_message) if obj.chat_message else "N/A"
    chat_message_safe.short_description = 'Chat Message'

    def error_message_safe(self, obj):
        return obj.error_message[:30] if obj.error_message else "No error message"
    error_message_safe.short_description = 'Error Message'

admin.site.register(ErrorLog, ErrorLogAdmin)


class AiGeneratedCodedmin(admin.ModelAdmin):
    list_display = ('chat_message',)
    search_fields = ('chat_message', 'code')


admin.site.register(Ai_generated_code, AiGeneratedCodedmin)
admin.site.register(UserloadedFile)
