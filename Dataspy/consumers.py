# # # consumers.py
# # import json
# # from channels.generic.websocket import AsyncWebsocketConsumer
# # from . tasks import send_notification
# # class ChatConsumer(AsyncWebsocketConsumer):
# #     async def connect(self):
# #         self.room_name = "chat_room"  # Room name or dynamic based on user
# #         self.room_group_name = f"chat_{self.room_name}"
# #
# #         # Join the room group
# #         await self.channel_layer.group_add(
# #             self.room_group_name,
# #             self.channel_name
# #         )
# #
# #         # Also subscribe to user-specific notifications if needed
# #         self.user_group_name = f"user_{self.scope['user'].id}"
# #         await self.channel_layer.group_add(
# #             self.user_group_name,
# #             self.channel_name
# #         )
# #
# #         await self.accept()
# #
# #     async def disconnect(self, close_code):
# #         # Leave both the room group and user group
# #         await self.channel_layer.group_discard(
# #             self.room_group_name,
# #             self.channel_name
# #         )
# #         await self.channel_layer.group_discard(
# #             self.user_group_name,
# #             self.channel_name
# #         )
# #
# #     async def receive(self, text_data):
# #         text_data_json = json.loads(text_data)
# #         user_message = text_data_json["text"]
# #
# #         # Send the message to the room group
# #         await self.channel_layer.group_send(
# #             self.room_group_name,
# #             {
# #                 "type": "chat_message",
# #                 "user": "user",
# #                 "text": user_message
# #             }
# #         )
# #
# #         # Optionally, you can call send_notification here as well
# #         # For example, if you want to send notifications for certain events
# #         # await send_notification(event_type="new_message", details=user_message, user_id=self.scope['user'].id)
# #
# #     # Receive message from room group
# #     async def chat_message(self, event):
# #         message = event["text"]
# #         user = event["user"]
# #
# #         # Send message to WebSocket
# #         await self.send(text_data=json.dumps({
# #             "user": user,
# #             "text": message
# #         }))
# #
# #     # This method can handle notification-specific messages
# #     async def notification_message(self, event):
# #         # Send notification to WebSocket
# #         await self.send(text_data=json.dumps({
# #             "event": event["event"],
# #             "details": event["details"]
# #         }))
# #
# #
# # # consumers.py
# # class NotificationConsumer(AsyncWebsocketConsumer):
# #     async def connect(self):
# #         self.user_group_name = f"user_{self.scope['user'].id}"
# #         # Join the user-specific notification group
# #         await self.channel_layer.group_add(
# #             self.user_group_name,
# #             self.channel_name
# #         )
# #
# #         await self.accept()
# #
# #     async def disconnect(self, close_code):
# #         await self.channel_layer.group_discard(
# #             self.user_group_name,
# #             self.channel_name
# #         )
# #
# #     # Receive notification from the group
# #     async def notification_message(self, event):
# #         # Send notification to WebSocket
# #         await self.send(text_data=json.dumps({
# #             "event": event["event"],
# #             "details": event["details"]
# #         }))
#
#
#
#
# # consumers.py
# import json
# from channels.generic.websocket import AsyncWebsocketConsumer
#
# class ChatConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         self.room_name = "chat_room"  # Room name or dynamic based on user
#         self.room_group_name = f"chat_{self.room_name}"
#
#         # Join the room group
#         await self.channel_layer.group_add(
#             self.room_group_name,
#             self.channel_name
#         )
#
#         # Also subscribe to user-specific notifications if needed
#         self.user_group_name = f"user_{self.scope['user'].id}"
#         await self.channel_layer.group_add(
#             self.user_group_name,
#             self.channel_name
#         )
#
#         await self.accept()
#
#     async def disconnect(self, close_code):
#         # Leave both the room group and user group
#         await self.channel_layer.group_discard(
#             self.room_group_name,
#             self.channel_name
#         )
#         await self.channel_layer.group_discard(
#             self.user_group_name,
#             self.channel_name
#         )
#
#     async def receive(self, text_data):
#         text_data_json = json.loads(text_data)
#         user_message = text_data_json["text"]
#
#         # Send the message to the room group
#         await self.channel_layer.group_send(
#             self.room_group_name,
#             {
#                 "type": "chat_message",
#                 "user": self.scope["user"].username,  # Assuming you want to send the user's username
#                 "text": user_message
#             }
#         )
#
#         # Optionally, trigger a notification asynchronously
#         if user_message:  # For example, send notification if a user sends a message
#             event_type = "new_message"
#             details = f"New message from {self.scope['user'].username}: {user_message}"
#             user_id = self.scope['user'].id
#             send_notification(event_type=event_type, details=details, user_id=user_id)  # Call the Celery task
#
#     async def chat_message(self, event):
#         message = event["text"]
#         user = event["user"]
#
#         # Send message to WebSocket
#         await self.send(text_data=json.dumps({
#             "user": user,
#             "text": message
#         }))
#
#     # This method can handle notification-specific messages
#     async def notification_message(self, event):
#         # Send notification to WebSocket
#         await self.send(text_data=json.dumps({
#             "event": event["event"],
#             "details": event["details"]
#         }))
