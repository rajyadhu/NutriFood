# Generated by Django 5.0.3 on 2024-04-01 08:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('foodapp', '0002_food_prediction'),
    ]

    operations = [
        migrations.AddField(
            model_name='registration',
            name='calory_needed',
            field=models.CharField(max_length=200, null=True),
        ),
        migrations.AddField(
            model_name='registration',
            name='height',
            field=models.CharField(max_length=200, null=True),
        ),
        migrations.AddField(
            model_name='registration',
            name='weight',
            field=models.CharField(max_length=200, null=True),
        ),
    ]