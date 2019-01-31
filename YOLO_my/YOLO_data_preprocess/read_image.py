
import json
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


import category_color_list as CCL

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
        #坐标
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
    
def readLabelFile(labelFileName):
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
    
def main():
    labelFileName = "./data_bdd/detection_val.json" #test
    infoListDict = readLabelFile(labelFileName)
    category2color = CCL.CreateColorList()
    for imageName in infoListDict.keys():
        imageName = "b1d0a191-2ed2269e.jpg"
        image = ReadImage(imageName,"./data_bdd/val")
        boxInfo = infoListDict[imageName]
        ShowImage(image,imageName,boxInfo,category2color)
        break
    #print (infoListDict)
if __name__ == '__main__':
    main()