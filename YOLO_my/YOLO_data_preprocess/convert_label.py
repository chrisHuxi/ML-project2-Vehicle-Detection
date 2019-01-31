
import json
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import category_color_list as CCL
import category_number_list as CNL

def ReadImage(imageName,imagePath):
    image = cv2.imread(imagePath+"//"+imageName, cv2.IMREAD_COLOR)
    # get b,g,r
    return image
    
def DrawBox(image,bboxList,category2color):
    imageLabeled = image[:]
    for box in bboxList:
        RGB = category2color[box[1]]
        BGR = (RGB[2],RGB[1],RGB[0])
        #RGB
        position = list(map(lambda x: int(x), box[0]))
        lineThikness = 4
        cv2.rectangle(imageLabeled, (position[0],position[1]), (position[2],position[3]), BGR, lineThikness)
        font = 0
        fontSize = 1
        fontThikness = 2
        (textWidth, textHeight) = cv2.getTextSize(box[1], font, fontSize, fontThikness)[0]
        cv2.rectangle(imageLabeled, (position[0]- int(lineThikness/2)  ,position[1] - textHeight - 3*lineThikness ), (position[0]+ textWidth ,position[1] ), BGR, cv2.FILLED)
        cv2.putText(imageLabeled, box[1], (position[0],position[1]-10), font, fontSize, (255,255,255), 2)
    return imageLabeled
    
def ShowImage(image,imageName,boxInfo,category2color):
    bboxList = boxInfo
    
    imageOriginal = image[:]
    imageOriginal = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    imageLabeled = DrawBox(image,bboxList,category2color)
    imageLabeled = cv2.cvtColor(imageLabeled, cv2.COLOR_BGR2RGB)

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(imageOriginal)
    plt.title('original')
    plt.subplot(122)
    plt.imshow(imageLabeled)
    plt.title('labeled')
    plt.show() 
    
def ReadLabelFile(labelFileName):
    frames = json.load(open(labelFileName, 'r'))
    currentImageName = frames[0]['name']
    infoListDict = {currentImageName:[]}
    for frame in frames:
        if frame['score'] != 1.0:
            print('attention') # all the scores are setted as 1
        if currentImageName == frame['name']:
            infoListDict[currentImageName].append([frame['bbox'],frame['category']])
        else:
            #new name
            currentImageName = frame['name']
            infoListDict[currentImageName] = [[frame['bbox'],frame['category']]]
    return infoListDict

#测试一下    
def RemoveBoxes(infoListDict,objectsShowAmount = 5):
    newInfoListDict = {}
    allBoxPosition = []
    listEveryLabel = {}
    sortedListEveryLabel = {}
    infoListDictKeySet = set(infoListDict.keys())
    #countImages = 0
    for everyImageName in infoListDictKeySet:
        #countImages += 1
        newInfoListDict[everyImageName] = []
        allBoxPosition = infoListDict[everyImageName][:]
        #[ [1.0,2.0,3.0,4.0], "car"],
        #  [[2.0,2.0,3.0,5.0], "car"],
        #  [[2.0,2.0,3.0,5.0], "person"] ]
        for i in allBoxPosition:
            if i[1] in set(listEveryLabel.keys()):
                listEveryLabel[i[1]].append(i)
            else:
                listEveryLabel[i[1]] = [i]
        for key in set(listEveryLabel.keys()):
            sortedListEveryLabel[key] = SortByDiagonal(listEveryLabel[key])
            #按照对角线长度来判别一个物体是否重要
        for key in set(sortedListEveryLabel.keys()):
            if len(sortedListEveryLabel[key]) < objectsShowAmount:
                newInfoListDict[everyImageName].extend(sortedListEveryLabel[key])
            else:
                sortedListEveryLabel[key] = sortedListEveryLabel[key][0:objectsShowAmount]
                newInfoListDict[everyImageName].extend(sortedListEveryLabel[key])
        #if (countImages%50 == 0):
        #    print(countImages)
    return newInfoListDict

def RemoveBoxes(infoListDict,objectsShowAmount = 5):
    newInfoListDict = {}
    allBoxPosition = []
    listEveryLabel = {}
    sortedListEveryLabel = {}
    infoListDictKeySet = set(infoListDict.keys())
    #countImages = 0
    for everyImageName in infoListDictKeySet:
        #countImages += 1
        newInfoListDict[everyImageName] = []
        allBoxPosition = infoListDict[everyImageName][:]
        #[ [1.0,2.0,3.0,4.0], "car"],
        #  [[2.0,2.0,3.0,5.0], "car"],
        #  [[2.0,2.0,3.0,5.0], "person"] ]
        for i in allBoxPosition:
            if i[1] in set(listEveryLabel.keys()):
                listEveryLabel[i[1]].append(i)
            else:
                listEveryLabel[i[1]] = [i]
        for key in set(listEveryLabel.keys()):
            sortedListEveryLabel[key] = SortByDiagonal(listEveryLabel[key])
            #按照对角线长度来判别一个物体是否重要
        for key in set(sortedListEveryLabel.keys()):
            if len(sortedListEveryLabel[key]) < objectsShowAmount:
                newInfoListDict[everyImageName].extend(sortedListEveryLabel[key])
            else:
                sortedListEveryLabel[key] = sortedListEveryLabel[key][0:objectsShowAmount]
                newInfoListDict[everyImageName].extend(sortedListEveryLabel[key])
        #if (countImages%50 == 0):
        #    print(countImages)
    return newInfoListDict

    
#把按类别分好类的list排序  
#输入：[ ['bbox':[1.0,2.0,3.0,4.0],'category':"car"],
#        ['bbox':[2.0,2.0,3.0,5.0],'category':"car"] ]  
def SortByDiagonal(classedList):
    sortedList = sorted(classedList,key = lambda classedList: np.linalg.norm(np.array([classedList[0][0],classedList[0][1]])-np.array([classedList[0][2],classedList[0][3]])), reverse = True)
    return sortedList

def CalculateArea(bbox4Point):
    return np.abs((bbox4Point[0]-bbox4Point[2])*(bbox4Point[1]-bbox4Point[3]))
    
    
def ChangeLabelType(bbox4Point,imageSize=[1280,720]):
    picture_width = imageSize[0]
    picture_height = imageSize[1]
    
    box_x_min = int(bbox4Point[0])  # 左上角横坐标
    box_y_min = int(bbox4Point[1])  # 左上角纵坐标
    box_x_max = int(bbox4Point[2])  # 右下角横坐标
    box_y_max = int(bbox4Point[3])  # 右下角纵坐标
    # 转成相对位置和宽高
    x_center = (box_x_min + box_x_max) / (2 * picture_width)
    y_center = (box_y_min + box_y_max) / (2 * picture_height)
    width = (box_x_max - box_x_min) / (picture_width)
    height = (box_y_max - box_y_min) / (picture_height)
    
    return [x_center,y_center,width,height]


#测试一下    
def main():
    labelFileName = "./data_bdd/detection_train.json" #test
    infoListDict = ReadLabelFile(labelFileName)
    print(len(infoListDict))
    #removedInfoListDict = RemoveBoxes(infoListDict, 4)
    return 
    
    print("-----------------")
    category2color = CCL.CreateColorList()
    category2number = CNL.CreateNumberList() 
    countImages = 0
    for imageName in infoListDict.keys():
        #image = ReadImage(imageName,"./data_bdd/train")
        removedInfoListDict = RemoveBoxes({imageName:infoListDict[imageName]}, 4)
        boxInfo = removedInfoListDict[imageName]
        with open("./converted/train_label/"+imageName[0:-3]+"txt","w") as f: 
            for box in boxInfo:
                darkNetText = str(category2number[box[1]]) + " "
                darkNetText += str(ChangeLabelType(box[0])[0]) + " " + str(ChangeLabelType(box[0])[1]) + " " + str(ChangeLabelType(box[0])[2]) + " " + str(ChangeLabelType(box[0])[3])
                f.write(darkNetText +'\n')
        countImages += 1
        if (countImages%50 == 0):
            print(countImages)

if __name__ == '__main__':
    main()
    #test = [ [[0,0,1280,720],"car"],[[2.0,2.0,3.0,5.0],"car"] ]  
    #sortedList = ChangeLabelType(test[0][0])
    #print(sortedList)