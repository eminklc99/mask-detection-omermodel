"""
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np


base_dir = r"C:/Users/Omer/PycharmProjects/pythonProject/data"
base_dirr = r"C:/Users/Omer/PycharmProjects/pythonProject/test"

IMAGE_SIZE=256
BATCH_SIZE=64

train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                              shear_range=0.2,
                                                              zoom_range=0.2,
                                                              horizontal_flip=True,
                                                              validation_split=0.1)

test_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                              validation_split=0.1)
train_datagen=train_datagen.flow_from_directory(base_dir,
                                                 target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                                 batch_size=BATCH_SIZE,
                                                 subset="training")
test_datagen=test_datagen.flow_from_directory(base_dirr,
                                                 target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                                 batch_size=BATCH_SIZE,
                                                 subset="validation")

model = keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=64,padding="same",strides=2,kernel_size=3,activation="relu",input_shape=(256,256,3)))
model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2,activation="sigmoid"))


model.summary()

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

model.fit(train_datagen,epochs=10,batch_size=64,validation_data=test_datagen)
model.save("omer.model")
"""





import tensorflow as tf
import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np


foto = tf.keras.preprocessing.image.load_img("31.jpeg", target_size=(256,256))		#test edilecek fotoyu yukledim
foto_matrisi = tf.keras.preprocessing.image.img_to_array(foto)				#test fotosunu matrise cevirdim
foto_matrisi = np.expand_dims(foto_matrisi, axis=0)
model = keras.models.load_model("omer.model")						#modeli yukledim
tahmin = model.predict(foto_matrisi)							#modele tahmin yaptim

if tahmin[0][0] == 1:
    print("MASKELİ ÖMER")
if tahmin[0][1] == 1:
    print("MASKESİZ ÖMER")
