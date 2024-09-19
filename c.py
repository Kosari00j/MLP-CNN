import os
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# تعیین مسیر دیتاست

data_dir = '/flower_images'


# تعیین اندازه تصاویر

img_height, img_width = 150, 150
batch_size = 32

# آماده‌کردن داده برای ورودی به CNN

train_data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_data_gen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = train_data_gen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')


# ساختن مدل CNN

model = models.Sequential([

    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(5, activation='softmax')
])


# کامپایل مدل

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# آموزش مدل

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=1)


# ذخیره مدل

model.save('flower_classification_model.h5')


# دریافت خطا و دقت از تاریخچه آموزش
train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']

# تعیین بازه دوره های آموزش
epochs = range(1, len(train_loss) + 1)

# رسم نمودار خطا
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b', label='خطای آموزش')
plt.plot(epochs, val_loss, 'r', label='خطای اعتبار سنجی')
plt.title('تغییر خطا در هر دوره')
plt.xlabel('دوره')
plt.ylabel('خطا')
plt.legend()

# رسم نمودار دقت
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'b', label='دقت آموزش')
plt.plot(epochs, val_acc, 'r', label='دقت اعتبار سنجی')
plt.title('تغییر دقت در هر دوره')
plt.xlabel('دوره')
plt.ylabel('دقت')
plt.legend()

# نمایش نمودار ها
plt.show()


