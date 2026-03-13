import cv2
import numpy as np
from tensorflow.keras.models import load_model
from focal_loss import BinaryFocalLoss
import matplotlib.pyplot as plt



def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


def iou_loss(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)  # Adăugați un epsilon mic pentru a evita împărțirea la zero
    return 1 - iou
    
model = load_model('unet_lstm\\lane_cv_dropout_batch.h5',custom_objects={'iou': iou,'iou_loss':iou_loss})

image_path = r'1.jpg'  
image = cv2.imread(image_path)
input_frame = cv2.resize(image, (512, 256))
input_frame = input_frame / 255.0
input_frame = np.expand_dims(input_frame, axis=0)


prediction = model.predict(input_frame)
prediction_resized = cv2.resize(np.squeeze(prediction[0]), (image.shape[1], image.shape[0]))


prediction_rgb = np.uint8(prediction_resized * 255)
prediction_rgb = cv2.cvtColor(prediction_rgb, cv2.COLOR_GRAY2RGB)

another_image_path = r'1.jpg'  
another_image = cv2.imread(another_image_path)


fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Imaginea de intrare')
axes[0].axis('off')

axes[1].imshow(prediction_rgb)
axes[1].set_title('Predictia')
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(another_image, cv2.COLOR_BGR2RGB))
axes[2].set_title('Imaginea filtrata')
axes[2].axis('off')

plt.show()