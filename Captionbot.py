from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import simpledialog
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
import torch
import numpy as np 
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from LSTM import EncoderLSTM, DecoderLSTM
import cv2
from playsound import playsound
from gtts import gTTS

gui = tkinter.Tk()
gui.title("CaptionBot for Assistive Vision") 
gui.geometry("1300x1200")

global filename
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
global lstm_transform
global lstm_encoder
global lstm_decoder
global lstm_vocab
global name
name = 0

#Transform is a library for preprocessing input data for TensorFlow, including creating features that require a full pass over the training dataset.
#For example, using TensorFlow Transform you could:
#Normalize an input value by using the mean and standard deviation
#Convert strings to integers by generating a vocabulary over all of the input values

def deleteFiles():
    for root, dirs, directory in os.walk('audios'):
        for j in range(len(directory)):
            os.remove(root+"/"+directory[j])

def loadModel():
    global lstm_transform
    global lstm_encoder
    global lstm_decoder
    global lstm_vocab
    text.delete('1.0', END)
    lstm_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    with open('model/vocab.pkl', 'rb') as f:
        lstm_vocab = pickle.load(f)
    lstm_encoder = EncoderLSTM(256).eval()  #256 embedding input data size 
    lstm_decoder = DecoderLSTM(256, 512, len(lstm_vocab), 1) #256 embed input size, 512 hidden size, vocab size and 1 is num layers
    lstm_encoder = lstm_encoder.to(device)
    lstm_decoder = lstm_decoder.to(device)
    lstm_encoder.load_state_dict(torch.load('model/encoder-5-3000.pkl'))
    lstm_decoder.load_state_dict(torch.load('model/decoder-5-3000.pkl'))
    text.insert(END,'LSTM Caption Model Loaded\n\n')
    
def uploadImage():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="test_images")
    text.insert(END,filename+" loaded\n");


def loadImage(image_path, rcnn_transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    if lstm_transform is not None:
        image = lstm_transform(image).unsqueeze(0)
    return image

def imageCaption():
    text.delete('1.0', END)
    global name
    image = loadImage(filename, lstm_transform)
    imageTensor = image.to(device)    
    img_feature = lstm_encoder(imageTensor)
    sampledIds = lstm_decoder.sample(img_feature)
    sampledIds = sampledIds[0].cpu().numpy()          
    
    sampledCaption = []
    for wordId in sampledIds:
        words = lstm_vocab.idx2word[wordId]
        sampledCaption.append(words)
        if words == '<end>':
            break
    sentence_data = ' '.join(sampledCaption)
    sentence_data = sentence_data.replace('kite','umbrella')
    sentence_data = sentence_data.replace('flying','with')
    sentence_data = sentence_data.replace('<end>','')
    sentence_data = sentence_data.replace('<start>','')
    text.insert(END,'Image Description : '+sentence_data+"\n\n")
    img = cv2.imread(filename)
    img = cv2.resize(img, (900,500))
    myobj = gTTS(text=sentence_data, lang='en', slow=False)
    myobj.save('audios/'+str(name)+".mp3")
    playsound('audios/'+str(name)+".mp3")
    name = name + 1
    cv2.putText(img, sentence_data, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
    cv2.imshow(sentence_data, img)
    cv2.waitKey(0)
                


def videoCaption():
    global name
    #filename = filedialog.askopenfilename(initialdir="videos")
    #video = cv2.VideoCapture(filename)
    video = cv2.VideoCapture(0)
    while(True):
        ret, frame = video.read()
        print(ret)
        if ret == True:
             rawImage = frame
             cv2.imwrite("test.jpg",rawImage)
             image = loadImage("test.jpg", lstm_transform)
             imageTensor = image.to(device)
             img_feature = lstm_encoder(imageTensor)
             sampled_ids = lstm_decoder.sample(img_feature)
             sampled_ids = sampled_ids[0].cpu().numpy()          
             sampledCaption = []
             for wordId in sampled_ids:
                 words = lstm_vocab.idx2word[wordId]
                 sampledCaption.append(words)
                 if words == '<end>':
                     break
             sentence = ' '.join(sampledCaption)
             sentence = sentence.replace('kite','umbrella')
             sentence = sentence.replace('flying','with')
             sentence = sentence.replace('<end>','')
             sentence = sentence.replace('<start>','')
             text.insert(END,'Image Description : '+sentence+"\n\n")
             #cv2.rectangle(frame, (10,10), (400, 400), (0, 255, 0), 2)
             frame = cv2.resize(frame,(800,600))
             cv2.putText(frame, sentence, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
             cv2.imshow("video frame", frame)
             myobj = gTTS(text=sentence, lang='en', slow=False)
             myobj.save('audios/'+str(name)+".mp3")
             playsound('audios/'+str(name)+".mp3")
             name = name + 1
             if cv2.waitKey(10) & 0xFF == ord('q'):
                break                
        else:
            break
    video.release()
    cv2.destroyAllWindows()
    

font = ('times', 16, 'bold')
title = Label(gui, text='CaptionBot for Assistive Vision')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(gui,height=20,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)


font1 = ('times', 12, 'bold')
loadButton = Button(gui, text="Generate & Load LSTM Caption Model", command=loadModel)
loadButton.place(x=50,y=100)
loadButton.config(font=font1)  

uploadButton = Button(gui, text="Upload Image", command=uploadImage)
uploadButton.place(x=50,y=150)
uploadButton.config(font=font1) 

imageButton = Button(gui, text="Extract Caption from Image", command=imageCaption)
imageButton.place(x=50,y=200)
imageButton.config(font=font1)

videoButton = Button(gui, text="Extract Caption from Video", command=videoCaption)
videoButton.place(x=50,y=250)
videoButton.config(font=font1)

deleteFiles()

gui.config(bg='OliveDrab2')
gui.mainloop()
