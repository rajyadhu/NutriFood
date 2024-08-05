from django.shortcuts import render, redirect
from django.http import HttpResponse
from . models import *
from django.contrib import messages
from django.contrib.auth.models import User, auth
from django.core.files.storage import FileSystemStorage
import os
from django.contrib.auth.hashers import make_password
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np
from keras.models import load_model
from django.http import HttpResponse, JsonResponse
from django.db.models import Q
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split



def home(request):
    return render(request, 'home.html')


def adminHome(request):
    return render(request, 'admin_home.html')


def userHome(request):
    return render(request, 'user_home.html')


def logout(request):
    hgt = Registration.objects.get(id = request.session['logg'])
    if Live_doctor.objects.filter(liv_dr = hgt).exists():
        Live_doctor.objects.filter(liv_dr = hgt).delete()
    auth.logout(request)
    return redirect('home')


def login(request):
    if request.method == 'POST':
        username = request.POST.get("user_name")
        password = request.POST.get("pword")
        user = auth.authenticate(username = username, password = password)
        if user is None:
            messages.success(request, 'Invalid credentials')
            return render(request, 'login.html')
        auth.login(request, user)
        if Registration.objects.filter(user = user, password = password).exists():
            logs = Registration.objects.filter(user = user, password = password)
            for value in logs:
                user_id = value.id
                usertype  = value.user_role
                if usertype == 'admin':
                    request.session['logg'] = user_id
                    return redirect('admin_home')

                elif usertype == 'user':
                    request.session['logg'] = user_id
                    return redirect('user_home')

                elif usertype == 'dietician':
                    request.session['logg'] = user_id
                    return redirect('dietician_home')

                else:
                    messages.success(request, 'Your access to the website is blocked. Please contact admin')
                    return redirect('login')
        else:
            messages.success(request, 'Username or password entered is incorrect')
            return redirect('login')
    else:
        return render(request, 'login.html')


def dietician_home(request):
    hgt = Registration.objects.get(id=request.session['logg'])
    if Live_doctor.objects.filter(liv_dr = hgt).exists():
        pass
    else:
        nyn = Live_doctor()
        nyn.liv_dr = hgt
        nyn.save()
    return render(request,'dietician_home.html')

def bmi(request):
    if request.method == 'POST':
        height = float(request.POST.get('height'))
        weight = float(request.POST.get('weight'))

        if height <= 0 or weight <= 0:
            return JsonResponse({'error': 'Height and weight must be positive numbers'})

        # BMI = weight in KG / (height in m * height in m)
        height_in_meters = height / 100
        bmi = weight / (height_in_meters * height_in_meters)

        status = ""
        if bmi < 18.5:
            status = "Underweight"
        elif bmi < 25:
            status = "Healthy"
        elif bmi < 30:
            status = "Overweight"
        else:
            status = "Obese"

        return JsonResponse({'bmi': bmi, 'status': status})

    return render(request, 'bmi.html')


def adminregister(request):
    if request.method == 'POST':
        lk = Registration.objects.all()
        for t in lk:
            if t.user_role == 'admin':
                messages.success(request, 'You are not allowed to be registered as admin')
                return redirect('home')
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')
        psw = request.POST.get('psw')
        reg1 = Registration.objects.all()
        for i in reg1:
            if i.user.email == email:
                messages.success(request, 'User already exists')
                return redirect('admin_reg')

        user_name = request.POST.get('user_name')
        for t in User.objects.all():
            if t.username == user_name:
                messages.success(request, 'Username taken. Please try another')
                return redirect('admin_reg')

        user = User.objects.create_user(username = user_name, email = email, password = psw, first_name = first_name, last_name = last_name)
        user.save()

        t = Registration()
        t.password = psw
        t.user_role = 'admin'
        t.user = user
        t.save()
        messages.success(request, 'You have successfully registered as admin')
        return redirect('home')
    else:

        return render(request, 'admin_reg.html')


def dietician_reg(request):
    if request.method == 'POST':
        imgg = request.FILES['imgg']
        fs = FileSystemStorage()
        fs.save(imgg.name,imgg)
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')
        psw = request.POST.get('psw')
        reg1 = Registration.objects.all()
        for i in reg1:
            if i.user.email == email:
                messages.success(request, 'User already exists')
                return redirect('dietician_reg')

        user_name = request.POST.get('user_name')
        for t in User.objects.all():
            if t.username == user_name:
                messages.success(request, 'Username taken. Please try another')
                return redirect('dietician_reg')

        user = User.objects.create_user(username = user_name, email = email, password = psw, first_name = first_name, last_name = last_name)
        user.save()

        t = Registration()
        t.password = psw
        t.user_role = 'dietician'
        t.user = user
        t.image = imgg
        t.save()
        messages.success(request, 'You have successfully registered as dietician')
        return redirect('home')
    else:

        return render(request, 'dietician_reg.html')


def userregister(request):
    if request.method == 'POST':
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')
        psw = request.POST.get('psw')
        hgt = request.POST.get('hgt')
        wgt = request.POST.get('wgt')
        gender = request.POST.get('gender')
        age = request.POST.get('age')
        lif_sty = request.POST.get('lif_sty')


        if lif_sty == 'sedentary':
            activity_factor = 1.2
        elif lif_sty == 'lightly active':
            activity_factor = 1.375
        elif lif_sty == 'moderately active':
            activity_factor = 1.55
        elif lif_sty == 'very active':
            activity_factor = 1.725
        else:
            activity_factor = 1.9


        bmr = 0
        if gender == 'Male':
            bmr = 88.362 + (13.397 * float(wgt)) + (4.799 * float(hgt)) - (5.677 * int(age))
        elif gender == 'Female':
            bmr = 447.593 + (9.247 * float(wgt)) + (3.098 * float(hgt)) - (4.330 * int(age))

        maintenance_calories = bmr * activity_factor


        reg1 = Registration.objects.all()
        for i in reg1:
            if i.user.email == email:
                messages.success(request, 'User already exists')
                return redirect('user_reg')

        user_name = request.POST.get('user_name')
        for t in User.objects.all():
            if t.username == user_name:
                messages.success(request, 'Username taken. Please try another')
                return redirect('user_reg')

        user = User.objects.create_user(username = user_name, email = email, password = psw, first_name = first_name, last_name = last_name)
        user.save()

        t = Registration()
        t.password = psw
        t.height = hgt
        t.weight = wgt
        t.gender = gender
        t.age = age
        t.life_style = lif_sty
        t.bmr = bmr
        t.calory_needed = maintenance_calories
        t.user_role = 'user'
        t.user = user
        t.save()
        messages.success(request, 'You have successfully registered as user')
        return redirect('home')
    else:
        return render(request, 'user_reg.html')


def up_pro_usr(request):
    gh = Registration.objects.get(id = request.session['logg'])
    rfy = gh.user.pk
    um = User.objects.get(id=rfy)
    if request.method == 'POST':
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')
        psw = request.POST.get('psw')
        hgt = request.POST.get('hgt')
        wgt = request.POST.get('wgt')
        gender = request.POST.get('gender')
        age = request.POST.get('age')
        lif_sty = request.POST.get('lif_sty')

        if lif_sty == 'sedentary':
            activity_factor = 1.2
        elif lif_sty == 'lightly active':
            activity_factor = 1.375
        elif lif_sty == 'moderately active':
            activity_factor = 1.55
        elif lif_sty == 'very active':
            activity_factor = 1.725
        else:
            activity_factor = 1.9

        bmr = 0
        if gender == 'Male':
            bmr = 88.362 + (13.397 * float(wgt)) + (4.799 * float(hgt)) - (5.677 * int(age))
        elif gender == 'Female':
            bmr = 447.593 + (9.247 * float(wgt)) + (3.098 * float(hgt)) - (4.330 * int(age))

        maintenance_calories = bmr * activity_factor

        user_name = request.POST.get('user_name')
        m = User.objects.all().exclude(username = um.username)
        for t in m:
            if t.username == user_name:
                messages.success(request, 'Username taken. Please try another')
                return redirect('up_pro_usr')

        passwords = make_password(psw)
        u = User.objects.get(id=rfy)
        u.password = passwords
        u.username = user_name
        u.email = email
        u.first_name = first_name
        u.last_name = last_name
        u.save()

        user = auth.authenticate(username=user_name, password=psw)
        auth.login(request, user)

        b = gh.id
        m = int(b)
        request.session['logg'] = m

        gh.password = psw
        gh.height = hgt
        gh.weight = wgt
        gh.gender = gender
        gh.age = age
        gh.life_style = lif_sty
        gh.bmr = bmr
        gh.calory_needed = maintenance_calories
        gh.user_role = 'user'
        gh.user = u
        gh.save()
        messages.success(request, 'You have updated your profile')
        return redirect('user_home')
    else:
        return render(request, 'up_pro_usr.html', {'gh':gh})


def fd_img_cla_usr(request):
    ggg = Food_prediction.objects.filter(food_reg = request.session['logg'])
    return render(request, 'fd_img_cla_usr.html', {'ggg': ggg})


def prdct_user(request):
    hgh = Registration.objects.get(id=request.session['logg'])
    if request.method == 'POST':
        imgg = request.FILES['img']
        fs = FileSystemStorage()
        file_path = fs.save(imgg.name, imgg)
        image_path = os.path.join(fs.location, file_path)

        data_train_path ="D:\\Nutri food\\foodfinal\\food\\foodapp\\training"
        img_width = 180
        img_height = 180
        data_train = tf.keras.utils.image_dataset_from_directory(
            data_train_path,
            shuffle=True,
            image_size=(img_width, img_height),
            batch_size=32,
            validation_split=False)

        data_category = data_train.class_names
        image = image_path
        image = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
        img_arr = tf.keras.utils.array_to_img(image)
        img_bat = tf.expand_dims(img_arr, 0)

        loaded_model = load_model('D:\\Nutri food\\foodfinal\\food\\foodapp\\food_Image_classifier_model.h5')

        predict = loaded_model.predict(img_bat)
        score = tf.nn.softmax(predict)
        print('Veg/Fruit in image is {} with accuracy of {:0.2f}'.format(data_category[np.argmax(score)],
                                                                         np.max(score) * 100))

        ressult = data_category[np.argmax(score)]

        wk = Food_prediction()
        wk.image = imgg
        wk.result = ressult
        wk.food_reg = hgh
        wk.save()
        messages.success(request, 'Prediction done successfully')
        return redirect('fd_img_cla_usr')

    return render(request, 'prdct_user.html')


def delete_predict_user(request, id):
    Food_prediction.objects.get(id = id).delete()
    messages.success(request, 'Prediction deleted successfully')
    return redirect('fd_img_cla_usr')


def fd_img_cla_adm(request):
    ggg = Food_prediction.objects.all()
    return render(request, 'fd_img_cla_adm.html', {'ggg': ggg})


def cr_new_diet_pln(request):
    gtg = Registration.objects.get(id = request.session['logg'])
    if request.method == 'POST':
        di_pln_for = request.POST.get('di_pln_for')
        if di_pln_for == 'm_w':
            calr = float(gtg.calory_needed)
        elif di_pln_for == 'l_w':
            calr = float(gtg.calory_needed) - 100
        else:
            calr = float(gtg.calory_needed) + 100


        # For monday
        mon_b = Diet_foods.objects.filter(Day = 'Monday')
        mon_b_c = Diet_foods.objects.filter(Day='Monday').count()
        tot_calr_m = 0
        for t in mon_b:
            gh = float(t.calory)
            tot_calr_m += gh

        if calr == tot_calr_m:
            for t in mon_b:
                t.quantity_needed = t.quantity
                t.protein_needed = t.protein
                t.carbohydrates_needed = t.carbohydrates
                t.fibre_needed = t.fibre
                t.fat_needed = t.fat
                t.current_needed_calory = t.calory
                t.save()

        elif calr < tot_calr_m:
            mon_b = Diet_foods.objects.filter(Day='Monday')
            needt = tot_calr_m - calr
            needt = needt/mon_b_c

            for t in mon_b:
                need_calry = float(t.calory) - float(needt)
                t.current_needed_calory = round(need_calry, 2)
                t.save()

            mon_b = Diet_foods.objects.filter(Day='Monday')
            for t in mon_b:
                need_qty = ( float(t.quantity) * float(t.current_needed_calory) ) / (float(t.calory))
                need_qty = float(need_qty)
                need_qty = round(need_qty, 2)
                t.quantity_needed = need_qty

                need_prot = (float(t.protein) * float(t.current_needed_calory)) / (float(t.calory))
                need_prot = float(need_prot)
                need_prot = round(need_prot, 2)
                t.protein_needed = need_prot

                need_carb = (float(t.carbohydrates) * float(t.current_needed_calory)) / (float(t.calory))
                need_carb = float(need_carb)
                need_carb = round(need_carb, 2)
                t.carbohydrates_needed = need_carb

                need_fib = ( float(t.fibre) * float(t.current_needed_calory) ) / ( float(t.calory) )
                need_fib = float(need_fib)
                need_fib = round(need_fib, 2)
                t.fibre_needed = need_fib

                need_fat = (float(t.fat) * float(t.current_needed_calory)) / (float(t.calory))
                need_fat = float(need_fat)
                need_fat = round(need_fat, 2)
                t.fat_needed = need_fat

                t.save()

        else:
            mon_b = Diet_foods.objects.filter(Day='Monday')
            needt = calr - tot_calr_m
            needt = needt / mon_b_c

            for t in mon_b:
                need_calry = float(t.calory) + float(needt)
                t.current_needed_calory = round(need_calry, 2)
                t.save()

            mon_b = Diet_foods.objects.filter(Day='Monday')
            for t in mon_b:
                need_qty = (float(t.quantity) * float(t.current_needed_calory)) / (float(t.calory))
                need_qty = float(need_qty)
                need_qty = round(need_qty, 2)
                t.quantity_needed = need_qty

                need_prot = (float(t.protein) * float(t.current_needed_calory)) / (float(t.calory))
                need_prot = float(need_prot)
                need_prot = round(need_prot, 2)
                t.protein_needed = need_prot

                need_carb = (float(t.carbohydrates) * float(t.current_needed_calory)) / (float(t.calory))
                need_carb = float(need_carb)
                need_carb = round(need_carb, 2)
                t.carbohydrates_needed = need_carb

                need_fib = (float(t.fibre) * float(t.current_needed_calory)) / (float(t.calory))
                need_fib = float(need_fib)
                need_fib = round(need_fib, 2)
                t.fibre_needed = need_fib

                need_fat = (float(t.fat) * float(t.current_needed_calory)) / (float(t.calory))
                need_fat = float(need_fat)
                need_fat = round(need_fat, 2)
                t.fat_needed = need_fat

                t.save()



        # For tuesday
        tue_b = Diet_foods.objects.filter(Day='Tuesday')
        tue_b_c = Diet_foods.objects.filter(Day='Tuesday').count()
        tot_calr_m = 0
        for t in tue_b:
            gh = float(t.calory)
            tot_calr_m += gh

        if calr == tot_calr_m:
            for t in tue_b:
                t.quantity_needed = t.quantity
                t.protein_needed = t.protein
                t.carbohydrates_needed = t.carbohydrates
                t.fibre_needed = t.fibre
                t.fat_needed = t.fat
                t.current_needed_calory = t.calory
                t.save()

        elif calr < tot_calr_m:
            tue_b = Diet_foods.objects.filter(Day='Tuesday')
            needt = tot_calr_m - calr
            needt = needt / tue_b_c

            for t in tue_b:
                need_calry = float(t.calory) - float(needt)
                t.current_needed_calory = round(need_calry, 2)
                t.save()

            tue_b = Diet_foods.objects.filter(Day='Tuesday')
            for t in tue_b:
                need_qty = (float(t.quantity) * float(t.current_needed_calory)) / (float(t.calory))
                need_qty = float(need_qty)
                need_qty = round(need_qty, 2)
                t.quantity_needed = need_qty

                need_prot = (float(t.protein) * float(t.current_needed_calory)) / (float(t.calory))
                need_prot = float(need_prot)
                need_prot = round(need_prot, 2)
                t.protein_needed = need_prot

                need_carb = (float(t.carbohydrates) * float(t.current_needed_calory)) / (float(t.calory))
                need_carb = float(need_carb)
                need_carb = round(need_carb, 2)
                t.carbohydrates_needed = need_carb

                need_fib = (float(t.fibre) * float(t.current_needed_calory)) / (float(t.calory))
                need_fib = float(need_fib)
                need_fib = round(need_fib, 2)
                t.fibre_needed = need_fib

                need_fat = (float(t.fat) * float(t.current_needed_calory)) / (float(t.calory))
                need_fat = float(need_fat)
                need_fat = round(need_fat, 2)
                t.fat_needed = need_fat

                t.save()

        else:
            tue_b = Diet_foods.objects.filter(Day='Tuesday')
            needt = calr - tot_calr_m
            needt = needt / tue_b_c

            for t in tue_b:
                need_calry = float(t.calory) + float(needt)
                t.current_needed_calory = round(need_calry, 2)
                t.save()

            tue_b = Diet_foods.objects.filter(Day='Tuesday')
            for t in tue_b:
                need_qty = (float(t.quantity) * float(t.current_needed_calory)) / (float(t.calory))
                need_qty = float(need_qty)
                need_qty = round(need_qty, 2)
                t.quantity_needed = need_qty

                need_prot = (float(t.protein) * float(t.current_needed_calory)) / (float(t.calory))
                need_prot = float(need_prot)
                need_prot = round(need_prot, 2)
                t.protein_needed = need_prot

                need_carb = (float(t.carbohydrates) * float(t.current_needed_calory)) / (float(t.calory))
                need_carb = float(need_carb)
                need_carb = round(need_carb, 2)
                t.carbohydrates_needed = need_carb

                need_fib = (float(t.fibre) * float(t.current_needed_calory)) / (float(t.calory))
                need_fib = float(need_fib)
                need_fib = round(need_fib, 2)
                t.fibre_needed = need_fib

                need_fat = (float(t.fat) * float(t.current_needed_calory)) / (float(t.calory))
                need_fat = float(need_fat)
                need_fat = round(need_fat, 2)
                t.fat_needed = need_fat

                t.save()




        # For wednesday
        wed_b = Diet_foods.objects.filter(Day='Wednesday')
        wed_b_c = Diet_foods.objects.filter(Day='Wednesday').count()
        tot_calr_m = 0
        for t in wed_b:
            gh = float(t.calory)
            tot_calr_m += gh

        if calr == tot_calr_m:
            for t in wed_b:
                t.quantity_needed = t.quantity
                t.protein_needed = t.protein
                t.carbohydrates_needed = t.carbohydrates
                t.fibre_needed = t.fibre
                t.fat_needed = t.fat
                t.current_needed_calory = t.calory
                t.save()

        elif calr < tot_calr_m:
            wed_b = Diet_foods.objects.filter(Day='Wednesday')
            needt = tot_calr_m - calr
            needt = needt / wed_b_c

            for t in wed_b:
                need_calry = float(t.calory) - float(needt)
                t.current_needed_calory = round(need_calry, 2)
                t.save()

            wed_b = Diet_foods.objects.filter(Day='Wednesday')
            for t in wed_b:
                need_qty = (float(t.quantity) * float(t.current_needed_calory)) / (float(t.calory))
                need_qty = float(need_qty)
                need_qty = round(need_qty, 2)
                t.quantity_needed = need_qty

                need_prot = (float(t.protein) * float(t.current_needed_calory)) / (float(t.calory))
                need_prot = float(need_prot)
                need_prot = round(need_prot, 2)
                t.protein_needed = need_prot

                need_carb = (float(t.carbohydrates) * float(t.current_needed_calory)) / (float(t.calory))
                need_carb = float(need_carb)
                need_carb = round(need_carb, 2)
                t.carbohydrates_needed = need_carb

                need_fib = (float(t.fibre) * float(t.current_needed_calory)) / (float(t.calory))
                need_fib = float(need_fib)
                need_fib = round(need_fib, 2)
                t.fibre_needed = need_fib

                need_fat = (float(t.fat) * float(t.current_needed_calory)) / (float(t.calory))
                need_fat = float(need_fat)
                need_fat = round(need_fat, 2)
                t.fat_needed = need_fat

                t.save()

        else:
            wed_b = Diet_foods.objects.filter(Day='Wednesday')
            needt = calr - tot_calr_m
            needt = needt / wed_b_c

            for t in wed_b:
                need_calry = float(t.calory) + float(needt)
                t.current_needed_calory = round(need_calry, 2)
                t.save()

            wed_b = Diet_foods.objects.filter(Day='Wednesday')
            for t in wed_b:
                need_qty = (float(t.quantity) * float(t.current_needed_calory)) / (float(t.calory))
                need_qty = float(need_qty)
                need_qty = round(need_qty, 2)
                t.quantity_needed = need_qty

                need_prot = (float(t.protein) * float(t.current_needed_calory)) / (float(t.calory))
                need_prot = float(need_prot)
                need_prot = round(need_prot, 2)
                t.protein_needed = need_prot

                need_carb = (float(t.carbohydrates) * float(t.current_needed_calory)) / (float(t.calory))
                need_carb = float(need_carb)
                need_carb = round(need_carb, 2)
                t.carbohydrates_needed = need_carb

                need_fib = (float(t.fibre) * float(t.current_needed_calory)) / (float(t.calory))
                need_fib = float(need_fib)
                need_fib = round(need_fib, 2)
                t.fibre_needed = need_fib

                need_fat = (float(t.fat) * float(t.current_needed_calory)) / (float(t.calory))
                need_fat = float(need_fat)
                need_fat = round(need_fat, 2)
                t.fat_needed = need_fat

                t.save()



        # For thursday
        thu_b = Diet_foods.objects.filter(Day='Thursday')
        thu_b_c = Diet_foods.objects.filter(Day='Thursday').count()
        tot_calr_m = 0
        for t in thu_b:
            gh = float(t.calory)
            tot_calr_m += gh

        if calr == tot_calr_m:
            for t in thu_b:
                t.quantity_needed = t.quantity
                t.protein_needed = t.protein
                t.carbohydrates_needed = t.carbohydrates
                t.fibre_needed = t.fibre
                t.fat_needed = t.fat
                t.current_needed_calory = t.calory
                t.save()

        elif calr < tot_calr_m:
            thu_b = Diet_foods.objects.filter(Day='Thursday')
            needt = tot_calr_m - calr
            needt = needt / thu_b_c

            for t in thu_b:
                need_calry = float(t.calory) - float(needt)
                t.current_needed_calory = round(need_calry, 2)
                t.save()

            thu_b = Diet_foods.objects.filter(Day='Thursday')
            for t in thu_b:
                need_qty = (float(t.quantity) * float(t.current_needed_calory)) / (float(t.calory))
                need_qty = float(need_qty)
                need_qty = round(need_qty, 2)
                t.quantity_needed = need_qty

                need_prot = (float(t.protein) * float(t.current_needed_calory)) / (float(t.calory))
                need_prot = float(need_prot)
                need_prot = round(need_prot, 2)
                t.protein_needed = need_prot

                need_carb = (float(t.carbohydrates) * float(t.current_needed_calory)) / (float(t.calory))
                need_carb = float(need_carb)
                need_carb = round(need_carb, 2)
                t.carbohydrates_needed = need_carb

                need_fib = (float(t.fibre) * float(t.current_needed_calory)) / (float(t.calory))
                need_fib = float(need_fib)
                need_fib = round(need_fib, 2)
                t.fibre_needed = need_fib

                need_fat = (float(t.fat) * float(t.current_needed_calory)) / (float(t.calory))
                need_fat = float(need_fat)
                need_fat = round(need_fat, 2)
                t.fat_needed = need_fat

                t.save()

        else:
            thu_b = Diet_foods.objects.filter(Day='Thursday')
            needt = calr - tot_calr_m
            needt = needt / wed_b_c

            for t in thu_b:
                need_calry = float(t.calory) + float(needt)
                t.current_needed_calory = round(need_calry, 2)
                t.save()

            thu_b = Diet_foods.objects.filter(Day='Thursday')
            for t in thu_b:
                need_qty = (float(t.quantity) * float(t.current_needed_calory)) / (float(t.calory))
                need_qty = float(need_qty)
                need_qty = round(need_qty, 2)
                t.quantity_needed = need_qty

                need_prot = (float(t.protein) * float(t.current_needed_calory)) / (float(t.calory))
                need_prot = float(need_prot)
                need_prot = round(need_prot, 2)
                t.protein_needed = need_prot

                need_carb = (float(t.carbohydrates) * float(t.current_needed_calory)) / (float(t.calory))
                need_carb = float(need_carb)
                need_carb = round(need_carb, 2)
                t.carbohydrates_needed = need_carb

                need_fib = (float(t.fibre) * float(t.current_needed_calory)) / (float(t.calory))
                need_fib = float(need_fib)
                need_fib = round(need_fib, 2)
                t.fibre_needed = need_fib

                need_fat = (float(t.fat) * float(t.current_needed_calory)) / (float(t.calory))
                need_fat = float(need_fat)
                need_fat = round(need_fat, 2)
                t.fat_needed = need_fat

                t.save()



        # For friday
        fri_b = Diet_foods.objects.filter(Day='Friday')
        fri_b_c = Diet_foods.objects.filter(Day='Friday').count()
        tot_calr_m = 0
        for t in fri_b:
            gh = float(t.calory)
            tot_calr_m += gh

        if calr == tot_calr_m:
            for t in fri_b:
                t.quantity_needed = t.quantity
                t.protein_needed = t.protein
                t.carbohydrates_needed = t.carbohydrates
                t.fibre_needed = t.fibre
                t.fat_needed = t.fat
                t.current_needed_calory = t.calory
                t.save()

        elif calr < tot_calr_m:
            fri_b = Diet_foods.objects.filter(Day='Friday')
            needt = tot_calr_m - calr
            needt = needt / fri_b_c

            for t in fri_b:
                need_calry = float(t.calory) - float(needt)
                t.current_needed_calory = round(need_calry, 2)
                t.save()

            fri_b = Diet_foods.objects.filter(Day='Friday')
            for t in fri_b:
                need_qty = (float(t.quantity) * float(t.current_needed_calory)) / (float(t.calory))
                need_qty = float(need_qty)
                need_qty = round(need_qty, 2)
                t.quantity_needed = need_qty

                need_prot = (float(t.protein) * float(t.current_needed_calory)) / (float(t.calory))
                need_prot = float(need_prot)
                need_prot = round(need_prot, 2)
                t.protein_needed = need_prot

                need_carb = (float(t.carbohydrates) * float(t.current_needed_calory)) / (float(t.calory))
                need_carb = float(need_carb)
                need_carb = round(need_carb, 2)
                t.carbohydrates_needed = need_carb

                need_fib = (float(t.fibre) * float(t.current_needed_calory)) / (float(t.calory))
                need_fib = float(need_fib)
                need_fib = round(need_fib, 2)
                t.fibre_needed = need_fib

                need_fat = (float(t.fat) * float(t.current_needed_calory)) / (float(t.calory))
                need_fat = float(need_fat)
                need_fat = round(need_fat, 2)
                t.fat_needed = need_fat

                t.save()

        else:
            fri_b = Diet_foods.objects.filter(Day='Friday')
            needt = calr - tot_calr_m
            needt = needt / fri_b_c

            for t in fri_b:
                need_calry = float(t.calory) + float(needt)
                t.current_needed_calory = round(need_calry, 2)
                t.save()

            fri_b = Diet_foods.objects.filter(Day='Friday')
            for t in fri_b:
                need_qty = (float(t.quantity) * float(t.current_needed_calory)) / (float(t.calory))
                need_qty = float(need_qty)
                need_qty = round(need_qty, 2)
                t.quantity_needed = need_qty

                need_prot = (float(t.protein) * float(t.current_needed_calory)) / (float(t.calory))
                need_prot = float(need_prot)
                need_prot = round(need_prot, 2)
                t.protein_needed = need_prot

                need_carb = (float(t.carbohydrates) * float(t.current_needed_calory)) / (float(t.calory))
                need_carb = float(need_carb)
                need_carb = round(need_carb, 2)
                t.carbohydrates_needed = need_carb

                need_fib = (float(t.fibre) * float(t.current_needed_calory)) / (float(t.calory))
                need_fib = float(need_fib)
                need_fib = round(need_fib, 2)
                t.fibre_needed = need_fib

                need_fat = (float(t.fat) * float(t.current_needed_calory)) / (float(t.calory))
                need_fat = float(need_fat)
                need_fat = round(need_fat, 2)
                t.fat_needed = need_fat

                t.save()



        # For saturday
        sat_b = Diet_foods.objects.filter(Day='Saturday')
        sat_b_c = Diet_foods.objects.filter(Day='Saturday').count()
        tot_calr_m = 0
        for t in sat_b:
            gh = float(t.calory)
            tot_calr_m += gh

        if calr == tot_calr_m:
            for t in sat_b:
                t.quantity_needed = t.quantity
                t.protein_needed = t.protein
                t.carbohydrates_needed = t.carbohydrates
                t.fibre_needed = t.fibre
                t.fat_needed = t.fat
                t.current_needed_calory = t.calory
                t.save()

        elif calr < tot_calr_m:
            sat_b = Diet_foods.objects.filter(Day='Saturday')
            needt = tot_calr_m - calr
            needt = needt / sat_b_c

            for t in sat_b:
                need_calry = float(t.calory) - float(needt)
                t.current_needed_calory = round(need_calry, 2)
                t.save()

            sat_b = Diet_foods.objects.filter(Day='Saturday')
            for t in sat_b:
                need_qty = (float(t.quantity) * float(t.current_needed_calory)) / (float(t.calory))
                need_qty = float(need_qty)
                need_qty = round(need_qty, 2)
                t.quantity_needed = need_qty

                need_prot = (float(t.protein) * float(t.current_needed_calory)) / (float(t.calory))
                need_prot = float(need_prot)
                need_prot = round(need_prot, 2)
                t.protein_needed = need_prot

                need_carb = (float(t.carbohydrates) * float(t.current_needed_calory)) / (float(t.calory))
                need_carb = float(need_carb)
                need_carb = round(need_carb, 2)
                t.carbohydrates_needed = need_carb

                need_fib = (float(t.fibre) * float(t.current_needed_calory)) / (float(t.calory))
                need_fib = float(need_fib)
                need_fib = round(need_fib, 2)
                t.fibre_needed = need_fib

                need_fat = (float(t.fat) * float(t.current_needed_calory)) / (float(t.calory))
                need_fat = float(need_fat)
                need_fat = round(need_fat, 2)
                t.fat_needed = need_fat

                t.save()

        else:
            sat_b = Diet_foods.objects.filter(Day='Saturday')
            needt = calr - tot_calr_m
            needt = needt / sat_b_c

            for t in sat_b:
                need_calry = float(t.calory) + float(needt)
                t.current_needed_calory = round(need_calry, 2)
                t.save()

            sat_b = Diet_foods.objects.filter(Day='Saturday')
            for t in sat_b:
                need_qty = (float(t.quantity) * float(t.current_needed_calory)) / (float(t.calory))
                need_qty = float(need_qty)
                need_qty = round(need_qty, 2)
                t.quantity_needed = need_qty

                need_prot = (float(t.protein) * float(t.current_needed_calory)) / (float(t.calory))
                need_prot = float(need_prot)
                need_prot = round(need_prot, 2)
                t.protein_needed = need_prot

                need_carb = (float(t.carbohydrates) * float(t.current_needed_calory)) / (float(t.calory))
                need_carb = float(need_carb)
                need_carb = round(need_carb, 2)
                t.carbohydrates_needed = need_carb

                need_fib = (float(t.fibre) * float(t.current_needed_calory)) / (float(t.calory))
                need_fib = float(need_fib)
                need_fib = round(need_fib, 2)
                t.fibre_needed = need_fib

                need_fat = (float(t.fat) * float(t.current_needed_calory)) / (float(t.calory))
                need_fat = float(need_fat)
                need_fat = round(need_fat, 2)
                t.fat_needed = need_fat

                t.save()



        # For sunday
        sun_b = Diet_foods.objects.filter(Day='Sunday')
        sun_b_c = Diet_foods.objects.filter(Day='Sunday').count()
        tot_calr_m = 0
        for t in sun_b:
            gh = float(t.calory)
            tot_calr_m += gh

        if calr == tot_calr_m:
            for t in sun_b:
                t.quantity_needed = t.quantity
                t.protein_needed = t.protein
                t.carbohydrates_needed = t.carbohydrates
                t.fibre_needed = t.fibre
                t.fat_needed = t.fat
                t.current_needed_calory = t.calory
                t.save()

        elif calr < tot_calr_m:
            sun_b = Diet_foods.objects.filter(Day='Sunday')
            needt = tot_calr_m - calr
            needt = needt / sun_b_c

            for t in sun_b:
                need_calry = float(t.calory) - float(needt)
                t.current_needed_calory = round(need_calry, 2)
                t.save()

            sun_b = Diet_foods.objects.filter(Day='Sunday')
            for t in sun_b:
                need_qty = (float(t.quantity) * float(t.current_needed_calory)) / (float(t.calory))
                need_qty = float(need_qty)
                need_qty = round(need_qty, 2)
                t.quantity_needed = need_qty

                need_prot = (float(t.protein) * float(t.current_needed_calory)) / (float(t.calory))
                need_prot = float(need_prot)
                need_prot = round(need_prot, 2)
                t.protein_needed = need_prot

                need_carb = (float(t.carbohydrates) * float(t.current_needed_calory)) / (float(t.calory))
                need_carb = float(need_carb)
                need_carb = round(need_carb, 2)
                t.carbohydrates_needed = need_carb

                need_fib = (float(t.fibre) * float(t.current_needed_calory)) / (float(t.calory))
                need_fib = float(need_fib)
                need_fib = round(need_fib, 2)
                t.fibre_needed = need_fib

                need_fat = (float(t.fat) * float(t.current_needed_calory)) / (float(t.calory))
                need_fat = float(need_fat)
                need_fat = round(need_fat, 2)
                t.fat_needed = need_fat

                t.save()

        else:
            sun_b = Diet_foods.objects.filter(Day='Sunday')
            needt = calr - tot_calr_m
            needt = needt / sun_b_c

            for t in sun_b:
                need_calry = float(t.calory) + float(needt)
                t.current_needed_calory = round(need_calry, 2)
                t.save()

            sun_b = Diet_foods.objects.filter(Day='Sunday')
            for t in sun_b:
                need_qty = (float(t.quantity) * float(t.current_needed_calory)) / (float(t.calory))
                need_qty = float(need_qty)
                need_qty = round(need_qty, 2)
                t.quantity_needed = need_qty

                need_prot = (float(t.protein) * float(t.current_needed_calory)) / (float(t.calory))
                need_prot = float(need_prot)
                need_prot = round(need_prot, 2)
                t.protein_needed = need_prot

                need_carb = (float(t.carbohydrates) * float(t.current_needed_calory)) / (float(t.calory))
                need_carb = float(need_carb)
                need_carb = round(need_carb, 2)
                t.carbohydrates_needed = need_carb

                need_fib = (float(t.fibre) * float(t.current_needed_calory)) / (float(t.calory))
                need_fib = float(need_fib)
                need_fib = round(need_fib, 2)
                t.fibre_needed = need_fib

                need_fat = (float(t.fat) * float(t.current_needed_calory)) / (float(t.calory))
                need_fat = float(need_fat)
                need_fat = round(need_fat, 2)
                t.fat_needed = need_fat

                t.save()
        return render(request, 'display_diet_pln_usr.html', {'calr': calr, 'mon_b': mon_b, 'tue_b':tue_b,
                                                             'wed_b':wed_b, 'thu_b':thu_b, 'fri_b':fri_b, 'sat_b':sat_b, 'sun_b':sun_b})

    return render(request, 'cr_new_diet_pln.html')


def cons_diet_usr(request):
    rtr = Registration.objects.filter(user_role = 'dietician')
    return render(request,'cons_diet_usr.html',{'rtr':rtr})


def m_m2(request, id):
    dtt = Registration.objects.get(id = id)
    request.session['selected_dietician'] = int(dtt.id)
    p = Registration.objects.get(id=request.session['logg'])
    bb = Messages.objects.filter(To_reg = p, From_reg = dtt)
    return render(request, 'msg2.html', {'bb': bb})



def already_sent_usr(request):
    dtt = Registration.objects.get(id = request.session['selected_dietician'])
    bb = Messages.objects.filter(To_reg = request.session['selected_dietician'])
    return render(request, 'already_sent_usr.html', {'bb': bb, 'dtt':dtt})


def del_msg_student(request,id):
    Messages.objects.get(id = id).delete()
    messages.success(request, 'Message deleted successfully')
    return redirect('already_sent_usr')


def reply_msg_student(request,id):
    pa = Messages.objects.get(id = id)
    toto = int(pa.From_reg.id)
    p_to = Registration.objects.get(id=toto)
    p = Registration.objects.get(id=request.session['logg'])
    if request.method == 'POST':
        msg_cont = request.POST.get('msg_cont')
        pa1 = Messages()
        pa1.Message_content = msg_cont
        pa1.From_reg = p
        pa1.To_reg = p_to
        pa1.save()
        messages.success(request, 'Message reply successful')
        redd = '/m_m2/' + str(p_to.id)
        return redirect(redd)
    return render(request,'reply_msg_student.html',{'pa':pa})


def sent_msg_student(request):
    kk = Registration.objects.filter(user_role='dietician')
    p = Registration.objects.get(id = request.session['logg'])
    ditc = Registration.objects.get(id=request.session['selected_dietician'])
    ditc = int(ditc.id)
    if request.method == 'POST':
        reg_to = Registration.objects.get(id=ditc)
        msg_cont = request.POST.get('msg_cont')
        nm = Messages()
        nm.Message_content = msg_cont
        nm.From_reg = p
        nm.To_reg = reg_to
        nm.save()
        messages.success(request, 'Message sent successfully')
        redd = '/m_m2/'+str(ditc)
        return redirect(redd)
    return render(request,'sent_msg_student.html',{'kk':kk,'ditc':ditc})


def m_m(request):
    p = Registration.objects.get(id = request.session['logg'])
    bb = Messages.objects.filter(To_reg = p)
    return render(request,'message.html',{'bb':bb})


def reply_msg_admin(request,id):
    pa = Messages.objects.get(id = id)
    toto = int(pa.From_reg.id)
    p_to = Registration.objects.get(id = toto)
    p = Registration.objects.get(id=request.session['logg'])
    if request.method == 'POST':
        msg_cont = request.POST.get('msg_cont')
        pa1 = Messages()
        pa1.Message_content = msg_cont
        pa1.From_reg = p
        pa1.To_reg = p_to
        pa1.save()
        messages.success(request, 'Message reply successful')
        return redirect('m_m')
    return render(request,'reply_msg_admin.html',{'pa':pa})


def sent_msg_admin(request):
    kk = Registration.objects.filter(user_role = 'user')
    p = Registration.objects.get(id = request.session['logg'])
    if request.method == 'POST':
        to_em = request.POST.get('to_em')
        ddp = int(to_em)
        reg_to = Registration.objects.get(id = ddp)
        msg_cont = request.POST.get('msg_cont')
        nm = Messages()
        nm.Message_content = msg_cont
        nm.From_reg = p
        nm.To_reg = reg_to
        nm.save()
        messages.success(request, 'Message sent successfully')
        return redirect('m_m')
    return render(request,'sent_msg_admin.html',{'kk':kk})


def already_sent_msg_diet(request):
    bb = Messages.objects.filter(From_reg = request.session['logg'])
    return render(request, 'already_sent_msg_diet.html', {'bb': bb})


def del_msg_admin(request,id):
    Messages.objects.get(id = id).delete()
    messages.success(request, 'Message deleted successfully')
    return redirect('already_sent_msg_diet')


def liv_chat_usr(request):
    tht = Live_doctor.objects.all()
    if request.method == 'POST':
        doc_sel = request.POST.get('doc_sel')
        dct = Registration.objects.get(id = doc_sel)

        mrt = Chat_message.objects.filter(ch_msg_doc = dct).count()
        if mrt >= 1:
            messages.success(request, 'Doctor is in another consultation. Please try after sometime.')
            return redirect('liv_chat_usr')

        request.session['liv_cht_doc'] = int(dct.id)
        return render(request, 'liv_chat_usr1.html', {'dct': dct})
    return render(request,'liv_chat_usr.html',{'tht':tht})


def send(request):
    tht = Registration.objects.get(id = request.session['logg'])
    tht1 = Registration.objects.get(id = request.session['liv_cht_doc'])
    message = request.POST.get('message')
    mkj = Chat_message()
    mkj.ch_messages = message
    mkj.ch_messages_reg_pat = int(tht.id)
    mkj.ch_messages_reg_pat1 = str(tht.user.first_name) +' '+str(tht.user.last_name)
    mkj.ch_msg_reg = tht
    mkj.ch_msg_doc = tht1
    mkj.from_person = str(tht.user.first_name)+' '+str(tht.user.last_name)
    mkj.save()
    return HttpResponse('Message sent successfully')


def getMessages(request):
    tht = Registration.objects.get(id = request.session['liv_cht_doc'])
    tht1 = Registration.objects.get(id = request.session['logg'])
    messages = Chat_message.objects.filter(ch_msg_doc = tht, ch_msg_reg = tht1)
    return JsonResponse({"messages": list(messages.values())})


def clr_cht(request):
    tht = Registration.objects.get(id = request.session['liv_cht_doc'])
    tht1 = Registration.objects.get(id = request.session['logg'])
    Chat_message.objects.filter(Q(ch_msg_reg = tht) | Q(ch_msg_reg = tht1)).delete()
    return redirect('liv_chat_usr')


def liv_chat_doct(request):
    return render(request,'liv_chat_doct.html')


def getMessages1(request):
    messages = Chat_message.objects.filter(ch_msg_doc = request.session['logg'])
    return JsonResponse({"messages": list(messages.values())})


def send1(request):
    message = request.POST.get('message')
    message1 = request.POST.get('message1')
    if message1:
        pass
    else:
        return HttpResponse('No patient messaged')
    message1 = int(message1)
    tht = Registration.objects.get(id=request.session['logg'])
    tht1 = Registration.objects.get(id=message1)
    mkj = Chat_message()
    mkj.ch_messages = message
    mkj.ch_messages_reg_pat = int(tht1.id)
    mkj.ch_messages_reg_pat1 = str(tht1.user.first_name) + ' ' + str(tht1.user.last_name)
    mkj.ch_msg_reg = tht1
    mkj.ch_msg_doc = tht
    mkj.from_person = str(tht.user.first_name) + ' ' + str(tht.user.last_name)
    mkj.save()
    return HttpResponse('Message sent successfully')


def clr_cht1(request):
    Chat_message.objects.filter(ch_msg_doc = request.session['logg']).delete()
    return redirect('liv_chat_doct')


def cr_model_chat_bott(request):
    # Load the dataset
    df = pd.read_csv('D:\\Nutri food\\foodfinal\\food\\foodapp\\FITNESS.csv',names=['Human', 'Assistant'])

    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['Human'])
    vocab_size = len(tokenizer.word_index) + 1

    # Convert text data to sequences and pad sequences
    sequences = tokenizer.texts_to_sequences(df['Human'])
    max_sequence_len = max([len(x) for x in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='post')

    # Prepare input and output data
    X = np.array(padded_sequences)
    Y = pd.get_dummies(df['Assistant']).values

    # Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Build the model
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_sequence_len))  # Increase embedding dimension
    model.add(LSTM(128, return_sequences=True))  # Increase LSTM units
    model.add(Dropout(0.5))  # Increase dropout rate
    model.add(LSTM(128))
    model.add(Dense(Y.shape[1], activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=400, batch_size=64)

    # Save the model
    model.save("D:\\Nutri food\\foodfinal\\food\\foodapp\\chat_model.h5")
    return HttpResponse('hello')


def cht_chat_bott(request):
    model = load_model("D:\\Nutri food\\foodfinal\\food\\foodapp\\chat_model.h5")



    df = pd.read_csv('D:\\Nutri food\\foodfinal\\food\\foodapp\\FITNESS.csv',
        names=['Human', 'Assistant'])

    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['Human'])

    # Convert text data to sequences and pad sequences
    sequences = tokenizer.texts_to_sequences(df['Human'])
    max_sequence_len = max([len(x) for x in sequences])



    def predict_answer(question):
        question_sequence = tokenizer.texts_to_sequences([question])
        padded_question = pad_sequences(question_sequence, maxlen=max_sequence_len, padding='post')
        predicted_prob = model.predict(padded_question)
        predicted_index = np.argmax(predicted_prob)
        predicted_answer = df['Assistant'][predicted_index]
        return predicted_answer

    print("Welcome to the Chatbot! Type 'quit' to exit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'quit':
            print("Chatbot: Goodbye!")
            break
        else:
            response = predict_answer(user_input)
            print("Chatbot:", response)

    return HttpResponse('hello')


def liv_chat_ai_assi(request):
    return render(request, 'liv_chat_ai_assi.html')


def send2(request):
    querry = request.POST.get('message')

    model = load_model("D:\\Nutri food\\foodfinal\\food\\foodapp\\chat_model.h5")

    df = pd.read_csv('D:\\Nutri food\\foodfinal\\food\\foodapp\\FITNESS.csv',names=['Human', 'Assistant'])

    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['Human'])

    # Convert text data to sequences and pad sequences
    sequences = tokenizer.texts_to_sequences(df['Human'])
    max_sequence_len = max([len(x) for x in sequences])

    def predict_answer(question):
        question_sequence = tokenizer.texts_to_sequences([question])
        padded_question = pad_sequences(question_sequence, maxlen=max_sequence_len, padding='post')
        predicted_prob = model.predict(padded_question)
        predicted_index = np.argmax(predicted_prob)
        predicted_answer = df['Assistant'][predicted_index]
        return predicted_answer



    user_input = querry.strip()

    response = predict_answer(user_input)
    print(response)


    return JsonResponse({"messages": response})
