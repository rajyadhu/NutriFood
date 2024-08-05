from django.db import models
from django.contrib.auth.models import User


class Registration(models.Model):
    password = models.CharField(max_length=200, null=True)
    image = models.ImageField(null=True)
    weight = models.CharField(max_length=200, null=True)
    height = models.CharField(max_length=200, null=True)
    age = models.CharField(max_length=200, null=True)
    gender = models.CharField(max_length=200, null=True)
    life_style = models.CharField(max_length=200, null=True)
    bmr = models.CharField(max_length=200, null=True)
    calory_needed = models.CharField(max_length=200, null=True)
    user_role = models.CharField(max_length=200, null=True)
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True)


class Food_prediction(models.Model):
    image = models.ImageField(null=True)
    result = models.CharField(max_length=200, null=True)
    food_reg = models.ForeignKey(Registration, on_delete=models.SET_NULL, null=True)


class Diet_foods(models.Model):
    Day = models.CharField(max_length=200, null=True)
    day_slot =  models.CharField(max_length=200, null=True)
    food_name = models.CharField(max_length=200, null=True)
    quantity = models.CharField(max_length=200, null=True)
    quantity_needed = models.CharField(max_length=200, null=True)
    unit = models.CharField(max_length=200, null=True)
    protein = models.CharField(max_length=200, null=True)
    protein_needed = models.CharField(max_length=200, null=True)
    carbohydrates = models.CharField(max_length=200, null=True)
    carbohydrates_needed = models.CharField(max_length=200, null=True)
    fibre = models.CharField(max_length=200, null=True)
    fibre_needed = models.CharField(max_length=200, null=True)
    fat = models.CharField(max_length=200, null=True)
    fat_needed = models.CharField(max_length=200, null=True)
    calory = models.CharField(max_length=200, null=True)
    current_needed_calory = models.CharField(max_length=200, null=True)


    def __str__(self):
        return self.Day


class Messages(models.Model):
    Message_content = models.TextField(null=True)
    From_reg = models.ForeignKey(Registration, on_delete=models.SET_NULL, null=True, related_name='from_message')
    To_reg = models.ForeignKey(Registration, on_delete=models.SET_NULL, null=True, related_name='to_message')


class Live_doctor(models.Model):
    liv_dr = models.ForeignKey(Registration, on_delete=models.CASCADE, null=True)


class Chat_message(models.Model):
    ch_messages = models.CharField(max_length = 600, null = True)
    ch_messages_reg_pat = models.CharField(max_length = 200, null = True)
    ch_messages_reg_pat1 = models.CharField(max_length=200, null=True)
    ch_msg_reg = models.ForeignKey(Registration, on_delete=models.SET_NULL, null=True, related_name='chat_patient')
    ch_msg_doc = models.ForeignKey(Registration, on_delete=models.SET_NULL, null=True, related_name='chat_doctor')
    from_person = models.CharField(max_length = 200, null = True)