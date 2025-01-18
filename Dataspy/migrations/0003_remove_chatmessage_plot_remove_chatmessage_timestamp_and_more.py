# Generated by Django 5.1.2 on 2024-12-19 01:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Dataspy', '0002_chatmessage_plot'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='chatmessage',
            name='plot',
        ),
        migrations.RemoveField(
            model_name='chatmessage',
            name='timestamp',
        ),
        migrations.AddField(
            model_name='chatmessage',
            name='matplot',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='chatmessage',
            name='plotly',
            field=models.TextField(blank=True, null=True),
        ),
    ]
