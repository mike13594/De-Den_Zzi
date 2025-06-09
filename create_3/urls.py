from django.urls import path
from create_3 import views

app_name = "create_3"
urlpatterns = [
    path("", views.create_3, name = "create_3"),
]