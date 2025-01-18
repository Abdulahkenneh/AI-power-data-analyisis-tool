from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
app_name = 'DataSpy'
urlpatterns = [
    # Dashboard
    path('', views.dashboard, name='dashboard'),
    
    
    path('userprofile/', views.userprofile_view, name='userprofile'),
    path('clean_data/', views.clean_data, name='clean_data'),
    path('spread_sheet/', views.display_spread_sheet, name='spread_sheet'),
    # Data Analysis URLs
    path('data_query/', views.data_query, name='data_query'),
    path('updata/', views.auto_updata, name='update'),
    path('api/task-status/<uuid:task_id>/',views.task_status, name='task_status'),
    path('tryer/',views.tryer,name='tryer'),
    path('delete-file/<int:id>/',views.del_file,name='delete'),
    path('insights/', views.insights_generation, name='insights'),
    path('code_generation/', views.code_generation, name='code_generation'),
    path('visualizations/', views.visualizations, name='visualizations'),

    # Reporting URLs
    path('reports/', views.reports, name='reports'),
    path('export/', views.export_options, name='export'),

    # Support URLs
    path('help/', views.help_documentation, name='help'),
    path('feedback/', views.feedback_community, name='feedback'),

    # User Settings
    path('user_settings/', views.user_settings, name='user_settings'),

    # Admin Panel (if user is an admin)
    path('admin_panel/', views.admin_panel, name='admin_panel'),
    path('custom_analysis/', views.custom_analysis, name='custom_analysis'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)