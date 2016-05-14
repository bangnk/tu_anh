from django.conf.urls import url

from views import HistView
from views import SiftView

urlpatterns = [
    url(r'^hists/?', HistView.as_view()),
    url(r'^$', SiftView.as_view()),
]