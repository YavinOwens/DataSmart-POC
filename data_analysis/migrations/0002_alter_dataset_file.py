# Generated by Django 4.2.17 on 2024-12-23 20:53

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('data_analysis', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='dataset',
            name='file',
            field=models.FileField(upload_to='datasets/', validators=[django.core.validators.FileExtensionValidator(allowed_extensions=['csv', 'xlsx', 'xls'])]),
        ),
    ]
