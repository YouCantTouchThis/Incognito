from django.db import models

# Create your models here.
class Diagnosis(models.Model):
    image = models.ImageField(upload_to = 'images/')
    email = models.EmailField()
