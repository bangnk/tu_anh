from django.conf.urls import url

from views import HistView

urlpatterns = [
    url(r'^$', HistView.as_view()),
]