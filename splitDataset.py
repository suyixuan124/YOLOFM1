import os
import random

trainval_percent = 0.9
train_percent = 0.9
xmlfilepath = 'D:/yolo/yoloxxx/yolov5/data/dataset/data/train/labels'
txtsavepath = 'data/ImageSets'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open('D:/yolo/yoloxxx/yolov5/data/txt/trainval.txt', 'w')
ftest = open('D:/yolo/yoloxxx/yolov5/data/txt/test.txt', 'w')
ftrain = open('D:/yolo/yoloxxx/yolov5/data/txt/train.txt', 'w')
fval = open('D:/yolo/yoloxxx/yolov5/data/txt/val.txt', 'w')

for i in list:

    name = 'D:/yolo/yoloxxx/yolov5/data/dataset/data/train/images/' + total_xml[i][:-4] + '.jpg' + '\n'
    # name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

    # name = 'E:/aimodelspaces/yolov7/data/images/' + total_xml[i][:-4] + '.jpg' + '\n'
    # name_0 = 'E:/aimodelspaces/yolov7/data/images/' + total_xml[i][:-4] + '.jpg' + '\n'
    # name_1 = 'E:/aimodelspaces/yolov7/data/images/' + total_xml[i][:-4] + '.jpg'
    # name_2 = 'E:/aimodelspaces/yolov7/data/images/' + total_xml[i][:-4] + '.txt'



    # if i in trainval:
    #
    #     ftest.write(name)
    #
    # else:
    #     ftrain.write(name)



ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
