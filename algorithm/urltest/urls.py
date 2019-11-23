from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^from_mainsys/$', views.from_mainsys),
    url(r'^docker_percent/$', views.docker_percent),
    url(r'^pred_power/$', views.pred_power),
    url(r'^pcvm_power/$', views.pcvm_power),
    url(r'^test/$', views.test),
    url(r'^cpu_freq/$', views.cpu_freq),
]
