from django.urls import path
from .views  import *


app_name = 'userlogs'
urlpatterns = [
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('register/', register, name='register'),
]
