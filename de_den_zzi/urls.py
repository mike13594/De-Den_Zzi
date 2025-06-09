"""
URL configuration for de_den_zzi project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.shortcuts import redirect
from de_den_zzi import views

app_name = "main"
urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.index_page, name = "index"),
    path("analysis/", include("analysis.urls")),
    path("classifier/", include("classifier_1.urls")),
    path("switch_sides/", include("switch_sides_2.urls")),
    path("create/", include("create_3.urls")),
]

urlpatterns += static(
    prefix=settings.MEDIA_URL,
    document_root=settings.MEDIA_ROOT,
)