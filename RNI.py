#!/usr/bin/python
# -*- coding: UTF-8 -*-
from PIL import Image
import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

NumOfData = 100
LengthOfPicture = 24
WidthOfPicture = 16
HiddenLayerNum = 20


def ImaPreprocess(num):
    imagename = 'data/' + str(num) + '.png'
    img = Image.open(imagename)
    img = img.convert('1')
    left, up, right, down = 0, 0, 0, 0
    length = img.size[1]
    width = img.size[0]
    for i in range(width):
        flag = 0
        for j in range(length):
            if img.getpixel((i, j)) != 255:
                left = i
                flag = 1
                break
        if flag == 1:
            break
    for i in range(width):
        flag = 0
        for j in range(length):
            if img.getpixel((width - 1 - i, j)) != 255:
                right = width - i
                flag = 1
                break
        if flag == 1:
            break
    for i in range(length):
        flag = 0
        for j in range(width):
            if img.getpixel((j, i)) != 255:
                up = i
                flag = 1
                break
        if flag == 1:
            break
    for i in range(length):
        flag = 0
        for j in range(width):
            if img.getpixel((j, length - 1 - i)) != 255:
                down = length - i
                flag = 1
                break
        if flag == 1:
            break
    region = (left, up, right, down)
    cropImg = img.crop(region)
    cropImg = cropImg.resize((WidthOfPicture, LengthOfPicture))
    # cropImg.save('M' + imagename)
    return cropImg


def RealImaPreprocess(num):
    imagename = str(num) + '.png'
    img = Image.open(imagename)
    img = img.convert('1')
    left, up, right, down = 0, 0, 0, 0
    length = img.size[1]
    width = img.size[0]
    for i in range(width):
        flag = 0
        for j in range(length):
            if img.getpixel((i, j)) != 255:
                left = i
                flag = 1
                break
        if flag == 1:
            break
    for i in range(width):
        flag = 0
        for j in range(length):
            if img.getpixel((width - 1 - i, j)) != 255:
                right = width - i
                flag = 1
                break
        if flag == 1:
            break
    for i in range(length):
        flag = 0
        for j in range(width):
            if img.getpixel((j, i)) != 255:
                up = i
                flag = 1
                break
        if flag == 1:
            break
    for i in range(length):
        flag = 0
        for j in range(width):
            if img.getpixel((j, length - 1 - i)) != 255:
                down = length - i
                flag = 1
                break
        if flag == 1:
            break
    region = (left, up, right, down)
    cropImg = img.crop(region)
    cropImg = cropImg.resize((WidthOfPicture, LengthOfPicture))
    # cropImg.save('M' + imagename)
    return cropImg
def GetImageMatrix(img):
    width = img.size[0]
    length = img.size[1]
    matrix = np.ones((length, width), dtype=np.uint8)
    for i in range(length):
        for j in range(width):
            matrix[i][j] = img.getpixel((j, i))
    return matrix

Num = np.zeros((NumOfData, LengthOfPicture, WidthOfPicture), dtype=np.uint8)
for i in range(NumOfData):
    img = ImaPreprocess(i)
    Num[i] = GetImageMatrix(img)

Label = np.zeros((NumOfData, 1, 10), dtype=np.uint8)
for i in range(NumOfData):
    trans = [0 for j in range(10)]
    trans[i % 10] = 1
    Label[i] = trans

# print Num[21], Label[21]
# print Num[38], Label[38]

net = buildNetwork(LengthOfPicture * WidthOfPicture, HiddenLayerNum, 10)
ds = SupervisedDataSet(LengthOfPicture * WidthOfPicture, 10)
for i in range(NumOfData):
    ds.addSample(Num[i].reshape((LengthOfPicture * WidthOfPicture, )), Label[i])
trainer = BackpropTrainer(net, ds)



def CheckTraining():
    print 'Training:',
    ErrorTrain = 0
    for i in range(NumOfData):
        # print net.activate(Num[i].reshape((LengthOfPicture * WidthOfPicture, )))
        # print net.activate(Num[i].reshape((LengthOfPicture * WidthOfPicture, ))).argmax(),
        # if i % 10 == 9:
        #     print '\n',
        if (i % 10 != net.activate(Num[i].reshape((LengthOfPicture * WidthOfPicture, ))).argmax()):
            ErrorTrain += 1
    print 1.0 - float(ErrorTrain) / NumOfData,
    return 1.0 - float(ErrorTrain) / NumOfData

# testimage = ImaPreprocess(12)
# testnum = GetImageMatrix(testimage)
# # print testnum
# print net.activate(testnum.reshape((864, )))
# print net.activate(testnum.reshape((864, ))).argmax()

def CheckValidation():
    print 'Validation: ',
    Error = 0
    for j in range(100, 130):
        testimage = ImaPreprocess(j)
        testnum = GetImageMatrix(testimage)
        # print testnum
        # print net.activate(testnum.reshape((LengthOfPicture * WidthOfPicture, )))
        # print j % 10
        # print net.activate(testnum.reshape((LengthOfPicture * WidthOfPicture, ))).argmax(),
        # if j % 10 == 9:
        #     print '\n',
        if (net.activate(testnum.reshape((LengthOfPicture * WidthOfPicture, ))).argmax() != j % 10):
            Error += 1
    print 1 - Error / 30.0,
    return 1 - Error / 30.0


i = 0
trainer.trainEpochs(epochs=100)
# print trainer.train()
CheckTraining()
CheckValidation()
# while(i < 10):
#     print i,
#     i += 1
#     a = CheckTraining()
#     b = CheckValidation()
#     print ' '
#     if(a < 0.95 or b < 0.90):
#         trainer.trainEpochs(epochs=20)
#         print trainer.train()
#     else:
#         break


testimage = RealImaPreprocess(1)
testnum = GetImageMatrix(testimage)
print '图片中的数字为：'
print net.activate(testnum.reshape((LengthOfPicture * WidthOfPicture, )))
print net.activate(testnum.reshape((LengthOfPicture * WidthOfPicture, ))).argmax()