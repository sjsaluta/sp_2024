from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader

def home(request):
    template = loader.get_template('home_space/home_space.html')
    context = {}
    return HttpResponse(template.render(context, request))

def about(request):
    template = loader.get_template('home_space/about.html')
    context = {}
    return HttpResponse(template.render(context, request))