from django.db import models

# Create your models here.

class UserAccount(models.Model):
    username = models.CharField(max_length=150, unique=True)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=128)  # Store hashed passwords in production!

    def __str__(self):
        return self.username
