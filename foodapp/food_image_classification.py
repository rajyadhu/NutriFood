

#prediction

# %%
image = 'download.jpg'
image = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image)
img_bat=tf.expand_dims(img_arr,0)

# %%
predict = model.predict(img_bat)

# %%
score = tf.nn.softmax(predict)

# %%
print('Veg/Fruit in image is {} with accuracy of {:0.2f}'.format(data_catogory[np.argmax(score)],np.max(score)*100))

# %%
# model.save('food_Image_classify.keras')
# Save the trained model
model.save('food_Image_classifier_model.h5')


