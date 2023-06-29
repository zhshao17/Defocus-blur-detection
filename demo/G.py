import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import cv2
from demo import test
import matplotlib.pyplot as plt
from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import LUHAONet
import torchvision.transforms as transforms

net = Lmodel.BR2Net()
stict = './model50.pth'
net.load_state_dict(torch.load(stict))
net = net.cuda()
net.eval()

name=0

flag=0
transf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def show():
    global name
    path1 = "./image/" + str(name - 1) + ".jpg"
    path2 = "./result/" + str(name - 1) + ".jpg"
    top1 = tk.Toplevel()
    image1 = Image.open(path1).resize((200, 200))
    image2 = Image.open(path2).resize((200, 200))
    img1 = ImageTk.PhotoImage(image1)
    img2 = ImageTk.PhotoImage(image2)
    canvas = tk.Canvas(top1, bg='white', height=240, width=420)
    # 定义图片1的位置
    image1 = canvas.create_image(5, 30, anchor='nw', image=img1)
    # 定义图片2的位置
    image = canvas.create_image(215, 30, anchor='nw', image=img2)
    canvas.pack()
    canvas.pack()
    top1.mainloop()

def get_img(filename, width, height):
    im = Image.open(filename).resize((width, height))
    im = ImageTk.PhotoImage(im)
    return im

def upload_file():
    global name,flag
    File = filedialog.askopenfilename()  # askopenfilename 1次上传1个；askopenfilenames1次上传多个
    # 此处File为上传的图片路径
    # 然后调用net函数，并且返回得到保存的图片的路径File_result
    img = cv2.imread(File)
    img = cv2.resize(img, (224, 224))
    path = "./image/" + str(name) + ".jpg"
    cv2.imwrite(path, img)
    img_tensor = transf(img).reshape(1, 3, 224, 224)
    test(model=net, input=img_tensor, name=name)
    print('over')
    name += 1
    flag=1

root = Tk()
root.title("散焦模糊检测")
root['width'] = 500
root['height'] = 400

Button(root, text="上传文件", width=14, command=upload_file).place(x=30, y=30)
Button(root, text="退出界面", width=14, command=root.destroy).place(x=190, y=30)
Button(root, text="给出结果", width=14, command=show).place(x=350,y=30)
#Button(root, text="给出结果", width=8, command=show).place(x=180, y=30)
yuanhui = get_img("yuanhui.jpg", 300, 300)
img_label2 = Label(root, text='', image=yuanhui)
img_label2.place(x=100, y=100, width=300, height=300)

root.mainloop()
print(11)