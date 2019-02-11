
def CreateColorList():
    categoryList = ['bus', 'traffic light', 'traffic sign', 'person', 'bike', 'truck', 'motor', 'car','train', 'rider'] 
    colorList = [ (25,24,22), (88,94,92), (186,40,53), (3,101,100), 
                  (248,252,137), (251,178,23), (153,80,84), (126,33,18),
                  (20,68,106),(90,13,67)
                  ]
    category2color = {}
    print(len(categoryList))
    print(len(colorList))
    
    for i in range(len(categoryList)):
        category2color[categoryList[i]] = colorList[i]
    return category2color
    
if __name__ == '__main__':
    CreateColorList()