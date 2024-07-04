# audio/models.py
from django.db import models
from django.utils import timezone


class AudioFile(models.Model):
    file = models.FileField(upload_to='uploads/')
    transcription = models.TextField(blank=True, null=True)
    summary = models.TextField(blank=True, null=True)
    uploaded_at = models.DateTimeField( default=timezone.now) 

    def __str__(self):
        return self.file.name
