from django.urls import path
from .views import TranscriptionView, SummarizationView

urlpatterns = [
    path('transcribe/', TranscriptionView.as_view(), name='transcribe'),
    path('summarize/', SummarizationView.as_view(), name='summarize'),
]
