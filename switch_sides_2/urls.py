from django.urls import path
from switch_sides_2 import views

app_name = "switch_sides_2"
urlpatterns = [
    path("", views.switch_sides_2, name = "switch_sides_2"),
]