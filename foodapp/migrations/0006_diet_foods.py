# Generated by Django 5.0.3 on 2024-04-02 06:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('foodapp', '0005_registration_bmr_registration_life_style'),
    ]

    operations = [
        migrations.CreateModel(
            name='Diet_foods',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Day', models.CharField(max_length=200, null=True)),
                ('day_slot', models.CharField(max_length=200, null=True)),
                ('food_name', models.CharField(max_length=200, null=True)),
                ('quantity', models.CharField(max_length=200, null=True)),
                ('unit', models.CharField(max_length=200, null=True)),
                ('protein', models.CharField(max_length=200, null=True)),
                ('carbohydrates', models.CharField(max_length=200, null=True)),
                ('fibre', models.CharField(max_length=200, null=True)),
                ('fat', models.CharField(max_length=200, null=True)),
                ('calory', models.CharField(max_length=200, null=True)),
            ],
        ),
    ]