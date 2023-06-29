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
import BR2Net

name = 0

net = LUHAONet.LUHAONet()
net.load_state_dict(torch.load("./LUHAONet.pth"))
net = net.cpu()
net.eval()


net2 = BR2Net.BR2Net()
net2.load_state_dict(torch.load("BR2Net.pth"))
net2 = net2.cpu()
net2.eval()


name = 0
flag = 0
transf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def show():
    global name
    path1 = "./image/" + str(name - 1) + ".jpg"
    path2 = "./result/1-" + str(name - 1) + ".jpg"
    path3 = "./result/2-" + str(name - 1) + ".jpg"
    top1 = tk.Toplevel()

    image1 = Image.open(path1).resize((200, 200))
    image2 = Image.open(path2).resize((200, 200))
    image3 = Image.open(path3).resize((200,200))
    img1 = ImageTk.PhotoImage(image1)
    img2 = ImageTk.PhotoImage(image2)
    img3 = ImageTk.PhotoImage(image3)
    image3 = Image.open(path3).resize((200, 200))

    canvas = tk.Canvas(top1, bg='white', height=250, width=630)
    #定义图片1的位置
    image1 = canvas.create_image(5, 20, anchor='nw', image=img1)
    # 定义图片2的位置
    image2 = canvas.create_image(215, 20, anchor='nw', image=img2)
    # 定义图片3的位置
    image3 = canvas.create_image(425, 20, anchor='nw', image=img3)

    canvas.pack()
    label1 = tk.Label(top1, text="原图", font=("微软雅黑", 8)).place(x=90, y=230)
    label2 = tk.Label(top1, text="LUHAONet", font=("微软雅黑",8)).place(x=290, y=230)
    label3 = tk.Label(top1, text="BR2Net", font=("微软雅黑", 8)).place(x=497, y=230)
    top1.mainloop()


def get_img(filename, width, height):
    im = Image.open(filename).resize((width, height))
    im = ImageTk.PhotoImage(im)
    return im


def upload_file():
    global name, flag, t
    text.delete(0.0, tk.END)
    text.insert(tk.INSERT, '     正在执行')
    text.update()
    File = filedialog.askopenfilename()  # askopenfilename 1次上传1个；askopenfilenames1次上传多个
    # 此处File为上传的图片路径
    # 然后调用net函数，并且返回得到保存的图片的路径File_result
    img = cv2.imread(File)
    img = cv2.resize(img, (224, 224))
    path = "./image/" + str(name) + ".jpg"
    cv2.imwrite(path, img)
    img_tensor = transf(img).reshape(1, 3, 224, 224)

    test(model=net, input=img_tensor, name=name, net = 1)
    test(model=net2, input=img_tensor, name=name, net=2)

    print('over')
    text.delete(0.0, tk.END)
    text.insert(tk.INSERT, '    执行完毕!')
    text.update()
    name += 1
    flag = 1
    show()


root = Tk()
root.title("散焦模糊检测")
root['width'] = 500
root['height'] = 450
DemoTitle = tk.Label(root, text="散焦模糊检测", font=("微软雅黑", 14)).place(x=190, y=10)
text = tk.Text(width=20, height=2)
text.place(x=175, y=360)
Button(root, text="上传文件", width=14, command=upload_file).place(x=30, y=400)
Button(root, text="退出界面", width=14, command=root.destroy).place(x=190, y=400)
Button(root, text="给出结果", width=14, command=show).place(x=350, y=400)

# Button(root, text="给出结果", width=8, command=show).place(x=180, y=30)
yuanhui = get_img("yuanhui.jpg", 300, 300)
img_label2 = Label(root, text='', image=yuanhui)
img_label2.place(x=100, y=50, width=300, height=300)

root.mainloop()
print(11)
