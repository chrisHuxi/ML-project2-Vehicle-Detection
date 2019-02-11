#还没测试
def CreateNumberList(versa = False):
    categoryList = ['bus', 'traffic light', 'traffic sign', 'person', 'bike', 'truck', 'motor', 'car','train', 'rider'] 
    numberList = range(10)
    if versa == False:
        category2number = {}
        for i in range(len(categoryList)):
            category2number[categoryList[i]] = numberList[i]
        return category2number
        
    elif versa == True:
        number2category = {}
        for i in range(len(categoryList)):
            number2category[numberList[i]] = categoryList[i]
        return number2category
    
    
if __name__ == '__main__':
    print(CreateNumberList())