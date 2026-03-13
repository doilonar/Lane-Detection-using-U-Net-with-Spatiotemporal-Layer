import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D,ReLU, MaxPooling2D, Conv2DTranspose, concatenate,Dropout,UpSampling2D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import os 
import matplotlib.pyplot as plt
from focal_loss import BinaryFocalLoss
import numpy as np




#iou metrics
def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    
    #Contraction path
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)


    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)


    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)


    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    
    conv5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv5)


    #Expansive path bin
    up6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4],axis=-1)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3],axis=-1)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)

    up8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2],axis=-1)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Dropout(0.1)(conv8)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)
    
    up9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1],axis=-1)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Dropout(0.1)(conv9)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9)



    
    bin_seg = Conv2D(1, (1, 1), activation='sigmoid',name='bin_seg')(conv9)

    
    model = Model(inputs=inputs, outputs=bin_seg)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3,decay=1e-3)
    tf.random.set_seed(40)
    
    model.compile(optimizer=optimizer, loss=BinaryFocalLoss(gamma=2),metrics=iou)
    return model

def image_mask_generator(image_folder, mask_folder, batch_size):
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,brightness_range=(0.1,2.0))
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    image_generator = image_datagen.flow_from_directory(
        image_folder,
        classes=os.listdir(image_folder),  # presupune că subfolderele au aceleași nume
        class_mode=None,
        color_mode='rgb',
        target_size=(256, 512),
        batch_size=batch_size,
        seed=42)
    
    mask_generator = mask_datagen.flow_from_directory(
        mask_folder,
        classes=os.listdir(mask_folder),  # presupune că subfolderele au aceleași nume
        class_mode=None,
        color_mode='grayscale',
        target_size=(256, 512),
        batch_size=batch_size,
        seed=42)
    
    while True:
        images = image_generator.next()
        masks = mask_generator.next()
        yield images, masks

# Configure your paths here
image_folder = r'lane'
mask_folder = r'lane_detect'
batch_size = 32

tf.config.threading.set_inter_op_parallelism_threads(3)
tf.config.threading.set_intra_op_parallelism_threads(3)

# Create a U-Net model
#model = keras.models.load_model('lane_detection_model.h5')
model = unet_model(input_size=(256, 512, 3))

# Create the generator
train_generator = image_mask_generator(image_folder, mask_folder, batch_size)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=r'tensorbord', histogram_freq=1)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='model_{epoch:02d}.h5', save_freq='epoch')
# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=32,
    epochs=100,
    shuffle=False,
    callbacks=[checkpoint_callback,tensorboard_callback]
    #,initial_epoch=1
    )

# Save the model
model.save('lane_cv_dropout_batch.h5')

plt.plot(history.history['iou'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('accuracy_plot.png')


# Plot and save loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss_plot.png')
