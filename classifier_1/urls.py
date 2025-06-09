from django.urls import path
from classifier_1 import views

app_name = "classifier_1"
urlpatterns = [
    path("", views.classifier_1, name = "classifier_1"),
]