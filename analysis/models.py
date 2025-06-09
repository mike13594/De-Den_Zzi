from django.db import models

# Create your models here.
class key_analysismodel(models.Model):
    political_name = models.CharField(max_length = 5)
    political_keyword = models.CharField(max_length = 30)
    political_issue = models.CharField(max_length = 30)
    issue_count = models.IntegerField()
    