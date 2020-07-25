from django.contrib import admin
from django.urls import path
from myapp.views import diagnosis_create_view
from django.conf import settings
from django.conf.urls.static import static 
urlpatterns = [
    path('', diagnosis_create_view),
]
if settings.DEBUG:
        urlpatterns += static(settings.MEDIA_URL,
                              document_root=settings.MEDIA_ROOT)
        urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)