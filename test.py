import tensorflow
import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import threading


import os
os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'
import cv2



emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D (128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D (pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout (0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense (1024, activation='relu'))
emotion_model.add(Dropout (0.5))
emotion_model.add(Dense (7, activation='softmax'))
emotion_model.load_weights ('model.h5')
cv2.ocl.setUseOpenCL (False)

emotion_dict={0:"   angry   ", 1:"  disgust  ", 2:"  Fear  ", 3:"  happy  ", 4:"   neutral  " , 5:"  sad  " , 6:" surprise" }

#cur_path = os.path.dirname(os.path.abspath(__file__))

emoji_dist = {
    0: r'C:\Users\vshas\OneDrive\Desktop\emotion_rec\emojis\angry.png',
    1: r'C:\Users\vshas\OneDrive\Desktop\emotion_rec\emojis\disgust.png',
    2: r'C:\Users\vshas\OneDrive\Desktop\emotion_rec\emojis\fear.png',
    3: r'C:\Users\vshas\OneDrive\Desktop\emotion_rec\emojis\happy.png',
    4: r'C:\Users\vshas\OneDrive\Desktop\emotion_rec\emojis\neutral.png',
    5: r'C:\Users\vshas\OneDrive\Desktop\emotion_rec\emojis\sad.png',
    6: r'C:\Users\vshas\OneDrive\Desktop\emotion_rec\emojis\surprise.png'
}



global last_frame1
last_frame1=np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text=[0]
global frame_number

def show_subject():

    cap1 = cv2.VideoCapture(r"C:\Users\vshas\Downloads\she1.mp4")
    if not cap1.isOpened():
       print("can't open camera")
    global frame_number
    length = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number +=1
    if frame_number >= length:
        exit()
    cap1.set(1, frame_number)
    flag1, frame1=cap1.read()
    frame1  = cv2.resize(frame1,(600,500))
    bounding_box=cv2.CascadeClassifier(r'C:\Users\vshas\OneDrive\Desktop\desktop\Unlock-Application-master\Haarcascades\haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0] = maxindex
    if flag1 is None:
        print("Major error!")
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk=ImageTk. PhotoImage(image=img)
        lmain.imgtk =imgtk
        lmain.configure(image=imgtk)
        root.update()
        lmain.after (10, show_subject)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

def show_avatar():
    frame2 = cv2.imread(emoji_dist[show_text[0]])
    #if frame2 is not None and not frame2.empty():
    pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(pic2)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2 = imgtk2
    lmain3.configure(text=emotion_dict[show_text[0]], font=('arial', 45, 'bold'))
    lmain2.configure(image=imgtk2)
    root.update()
    lmain2.after(10, show_avatar)
    #else:
     #   print("Invalid frame or empty image")
        # Handle the error, e.g., display a default image or show an error message    
    
if __name__ == '__main__':
    frame_number = 0
    root = tk.Tk()
    root.title("photo to emoji")
    root.geometry("1400x900+100+10")
    root['bg'] = 'black'

    lmain = tk.Label(master=root, padx=50, bd=10)
    lmain2 = tk.Label(master=root, bd=10)
    lmain3 = tk.Label(master=root, bd=10, fg="#CDCDCD", bg='black')
    lmain.pack(side=tk.LEFT)
    lmain.place(x=50, y=250)
    lmain3.pack()
    lmain3.place(x=960, y=250)
    lmain2.pack(side=tk.RIGHT)
    lmain2.place(x=900, y=350)

    exitButton = Button(root, text='Quit', fg='red', command=root.destroy, font=("arial", 25, 'bold'))
    exitButton.pack(side=tk.BOTTOM)

    # Start your functions after initializing the Tkinter root window
    threading.Thread(target=show_subject).start()
    threading.Thread(target=show_avatar).start()

    # Start the Tkinter main loop
    root.mainloop()



