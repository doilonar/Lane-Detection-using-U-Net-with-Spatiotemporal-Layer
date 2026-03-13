import cv2
import numpy as np
from tensorflow.keras.models import load_model
from focal_loss import BinaryFocalLoss
from tensorflow.keras.utils import plot_model

input_video_path = r'curved_lane.mp4'  
cap = cv2.VideoCapture(input_video_path)


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))



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
    iou = (intersection + 1e-7) / (union + 1e-7)  
    return 1 - iou

model = load_model('unet_lstm\\lane_cv_dropout_batch.h5',custom_objects={'iou': iou,'iou_loss':iou_loss})
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
print(model.summary())
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    input_video = frame

    input_frame = cv2.resize(frame, (512, 256))


    input_frame = input_frame / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)


    prediction = model.predict(input_frame)



    prediction_resized = cv2.resize(np.squeeze(prediction[0]), (width, height))
    

    prediction_rgb = cv2.cvtColor(prediction_resized, cv2.COLOR_GRAY2BGR)
    prediction_rgb = np.uint8(prediction_rgb * 255)
    
    window1 = cv2.addWeighted(prediction_rgb, 1, input_video, 1, 0)

    cv2.imshow('Prediction', window1)




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()