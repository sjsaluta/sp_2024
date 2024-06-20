from .models import *
from django import forms

class VideoForm(forms.Form):
    name= forms.CharField(max_length=500)
    videofile= forms.FileField()

class PredictForm(forms.Form):
    videofile= forms.FileField()
