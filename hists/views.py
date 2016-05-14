from django.shortcuts import render
from django.http import JsonResponse
from django.views.generic import TemplateView
from models import SearchModel
from django.views.decorators.cache import never_cache



class HistView(TemplateView):
    def get(self, request, *args, **kwargs):
        return render(request, 'index.html')

    @never_cache
    def post(self, request, *args, **kwargs):
        model = SearchModel()
        result = model.sdd(request.FILES['image'])
        return JsonResponse({'result': result})


class SiftView(TemplateView):
    def get(self, request, *args, **kwargs):
        return render(request, 'index.html')

    @never_cache
    def post(self, request, *args, **kwargs):
        model = SearchModel()
        result = model.cosine_sift(request.FILES['image'])
        return JsonResponse({'result': result})
