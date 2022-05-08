import cv2 # 导入
from train import emotion_analysis, reshape_dataset
from model import build_model
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
os.environ["KMP _DUPLICATE_LIB_OK"]="TRUE"
from PIL import Image
capture = cv2.VideoCapture(0)
cv2.namedWindow('摄像头')

num_classes = 3
model = load_model('./model_checkpoints/rps.h5')

while(capture.isOpened()):
    ret, frame = capture.read()
    


    #img = Image.fromarray(img)
    img = cv2.resize(frame,(150, 150))
    x_train = np.array(img)
    #x_train /= 255
    x_train = x_train.reshape(1, 150, 150, 3)
    #x_train = x_train.astype('float32')
    # x = np.array(img)
    # x = np.expand_dims(x, axis=0)
    #
    # #x /= 255
    objects = ('paper','rock','scissors')

    custom = model.predict(x_train)
    ans = np.argmax(custom[0])

    frame = cv2.putText(frame, objects[ans]+' '+str(custom[0][ans]*100)+'%', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (55,255,155), 2)
    #显示窗口第一个参数是窗口名，第二个参数是内容
    cv2.imshow('emotion', frame)
    if cv2.waitKey(1) == ord('q'):#按q退出
        break
capture.release()
cv2.destroyAllWindows()


#while True:
#
#     ret, frame = capture.read() # 读取视频图片
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # 灰度
#
#     faces = face.detectMultiScale(gray,1.1,3,0,(100,100))
#
#     for (x, y, w, h) in faces: # 5个参数，一个参数图片 ，2 坐标原点，3 识别大小，4，颜色5，线宽
#
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#         cv2.imshow('摄像头', frame) # 显示
#
#     if cv2.waitKey(5) & 0xFF == ord('q'):
#
#         break
# capture.release() # 释放资源
#
# cv2.destroyAllWindows() # 关闭窗口