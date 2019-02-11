import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import category_color_list as CCL
import category_number_list as CNL
#假设我们得到了一个7*7*(5+5+20) 的tensor, 
#需要根据这个部分完成prediction的部分

# templet
#input   : 
#output  : 
#function:
#def 


'''
https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/
https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.g137784ab86_4_3745
https://www.jianshu.com/p/23295376c44d

loss function:
https://github.com/experiencor/keras-yolo2/blob/master/Yolo%20Step-by-Step.ipynb
'''

class Bbox:   #bbox_number = 2, class_number = 20
    #_bbox_index = (i,j,m)
    #_coord = [x_center,y_center,width,height]
    #_confidence = float()
    #_prob_array = np.array([0]*20)
    #_class_index = -1  为对应的类别数值，初始化为-1
    def __init__(self,bbox_index_tuble,coord,confidence,prob_array):
        self.bbox_index = bbox_index_tuble
        self.coord = coord
        self.confidence = confidence
        self.prob_array = prob_array
        self.class_index = -1
    def _DisplayInfo(self):
        print("this bbox index:  " + str(self._bbox_index))
        print("this bbox coordinate:  " + str(self._coord))
        print("confidence: " + str(self._confidence))
        print("prob array: " + str(self._prob_array))
        
#input   :  print_object打印对象，print_titel打印标题，test_print_flag是否打印 
#output  :  无，打印到屏幕
#function: 格式化打印输出测试信息
test_print_flag = True
def TestPrint(print_object,print_titel,print_flag = test_print_flag):
    if print_flag == True:
        print("++++++++++++ ----->")
        print(print_titel+" : ")
        print(print_object)
        print("------------ <----")
    else:
        pass


        
#input   : eg.7*7*(5+5+20) 的tensor ==> cell_row_len * cell_column_len * (5 * bbox_number + class_number)
#output  : eg.98个bbox对象 ==> cell_row_len * cell_column_len * bbox_number 个 bbox
#function: 将从网络得到的数据转化为bbox的list
def GetBbox(nn_output, grid_information_dict):
    cell_row_len = grid_information_dict["cell_row_len"]
    cell_column_len = grid_information_dict["cell_column_len"]
    bbox_number = grid_information_dict["bbox_number"]
    class_number = grid_information_dict["class_number"]
    bbox_list = []
    nn_output = nn_output.reshape((cell_row_len,cell_column_len,-1))
    TestPrint(nn_output,"nn_output")
    for i in range(cell_row_len):
        for j in range(cell_column_len):
            for m in range(bbox_number):
                prob = np.array(nn_output[i,j,-class_number:])
                coordinate = nn_output[i,j,0:4]
                confidence = nn_output[i,j,m*5 + 4]
                bbox = Bbox((i,j,m),coordinate,confidence,prob)
                bbox_list.append(bbox)
    return bbox_list

#input   : 由nn输出的结果重新排列后得到的bbox_list, 以及要设置的threshold 越高则被检测的物体越少
#output  : 经过NMS后的new_bbox
#function: get class scores ,set threshold, sort descending, NMS    
def Process(bbox_list, threshold):
    #step1. 将confidence乘以class prob 得到class scores
    class_scores = []
    for bbox in bbox_list:
        class_scores.append(bbox.confidence * bbox.prob_array)
    class_scores = np.array(class_scores).T
    
    #变成了一个20 * 98的数组的array
    TestPrint(class_scores,"scores")
    TestPrint(class_scores[2],"scores example")

    #step2. set threshold
    class_scores_thres = (class_scores < threshold) * class_scores
    
    #变成了一个很多0的20 * 98的数组的array
    TestPrint(class_scores_thres,"scores thresholded")
    TestPrint(class_scores_thres[2],"scores example thresholded")
    
    #step3. sort 这部分需要特别注意，逻辑上别出问题了
    for row_index in range(class_scores_thres.shape[0]):
        sort_index = class_scores_thres[row_index].argsort() #按行的值，对列排序

        class_scores_sorted = class_scores_thres[:,sort_index]
        row = class_scores_sorted[row_index]
        
        row_sorted_remove_max = row[:] #做mask来用
        
        #第一个最大值索引必然是0
        bbox_max_index = 0
            
        #对每一个最大值
        while(1):
            #如果最大值已经是0了，或者最大值的指针到了最后一位了，则跳出循环
            if row[bbox_max_index] == 0.0 or bbox_max_index == len(row) - 1:
                break
                
            bbox_cur_point = bbox_max_index    
            
            before_sort_bbox_index_max = np.where(sort_index == bbox_max_index)[0][0] #应该是唯一索引，所以加[0][0]
             
            for i in range(bbox_max_index+1,len(row)):
                #class score最大的列索引 => 通过 sort_index
                #得到 排序前的索引 => 通过bbox找到对应bbox
                    
                if row[bbox_cur_point] == 0.0:
                    bbox_cur_point += 1
                    break
                else:
                    before_sort_bbox_index_cur = np.where(sort_index == bbox_cur_point)[0][0]
                        
                    IoU = CalculateIoU(bbox_list[before_sort_bbox_index_max],bbox_list[before_sort_bbox_index_cur])
                    if IoU > 0.5:
                        class_scores_sorted[row_index][bbox_cur_point] = 0.0
                        
                    bbox_cur_point += 1

            row_sorted_remove_max[bbox_max_index] = 0.0
                
            bbox_max_index = row_sorted_remove_max.argmax()
            #去掉最大值后的最大值(即第二个最大值)
    
    new_bbox_list = []
    
    #利用最后一次排序的索引找到对应的bbox
    for i in range(class_scores_sorted.shape[1]):   #98
        max_class_id = class_scores_sorted[:,i].argmax()
        if class_scores_sorted[:,i][max_class_id] > 0:
            #如果说一个bbox对应的列里面还有不为0的最大值，找到这个最大值，以及这个bbox在原来list里的位置
            before_sort_bbox_index = np.where(sort_index == i)[0][0]
            #print(before_sort_bbox_index[0][0])
            bbox = bbox_list[before_sort_bbox_index]
            
            #把这个bbox的class_index属性赋值为score最大的行数
            bbox.class_index = max_class_id
            new_bbox_list.append(bbox)
    
    return new_bbox_list
    


#input   : coord: [x_center,y_center,w,h](相对值), image size
#output  : coord: [x1,y1,x2,y2](绝对值)
#function: [x_center,y_center,h,w] => [x1,y1,x2,y2]
def CoordinateTransform(coord,image_size):
    picture_width = image_size[0]
    picture_height = image_size[1]
    x_center,y_center,h,w = coord
    x1 = int((x_center - w/2) * picture_width)
    x2 = int((x_center + w/2) * picture_width)
    y1 = int((y_center - h/2) * picture_height)
    y2 = int((y_center + h/2) * picture_height)
    
    coord_abs = [x1,y1,x2,y2]
    return coord_abs

#input   : 两个bbox对象
#output  : IoU_value
#function: 计算两个bbox的IoU值
def CalculateIoU(bbox1, bbox2):
    x1_bbox1, y1_bbox1, x2_bbox1, y2_bbox1 = CoordinateTransform(bbox1.coord,[1000,1000])
    x1_bbox2, y1_bbox2, x2_bbox2, y2_bbox2 = CoordinateTransform(bbox2.coord,[1000,1000])
    
    area_bbox1 = (x2_bbox1-x1_bbox1)*(y2_bbox1-y1_bbox1)
    area_bbox2 = (x2_bbox2-x1_bbox2)*(y2_bbox2-y1_bbox2)

    xx1 = max(x1_bbox1, x1_bbox2)
    yy1 = max(y1_bbox1, y1_bbox2)
    xx2 = min(x2_bbox1, x2_bbox2)
    yy2 = min(y2_bbox1, y2_bbox2)

    h = max(0, yy2-yy1)
    w = max(0, xx2-xx1)

    intersection = w * h

    IoU_value = intersection / (area_bbox1 + area_bbox2 - intersection + 1e-6)

    return IoU_value
    
#input   : image_name 图片名称（test set里）,image_path 路径
#output  : 读入的图片
#function: 读入图片
def ReadImage(image_name,image_path):
    image = cv2.imread(image_path+"//"+image_name, cv2.IMREAD_COLOR)
    return image    

#input   : image 原图, image_name 图片名称, bbox_list 对象列表, category2color 类别颜色对应列表, number2category 数字类别对应列表
#output  : 无 输出为图片
#function: 将原图和标记好的图（预测的图）打印在一起
def ShowImage(image,image_name,bbox_list,category2color,number2category,save_flag = True):
    image_original = image[:]
    image_original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image_labeled = DrawBox(image,bbox_list,category2color,number2category)
    image_labeled = cv2.cvtColor(image_labeled, cv2.COLOR_BGR2RGB)

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(image_original)
    plt.title('original')
    plt.subplot(122)
    plt.imshow(image_labeled)
    plt.title('labeled')
    if save_flag == True:
        plt.show() 
        plt.savefig(".\test_result\\"+image_name)
    else:
        plt.show() 
    
#input   : image, 经过处理的bbox_list(每个元素都是一个对象), category2color 类别颜色对应列表, number2category 数字类别对应列表
#output  : 画好bbox的图
#function: 根据bbox的信息在图上画框
def DrawBox(image,bbox_list,category2color,number2category):
    image_labeled = image[:]
    for box in bbox_list:
        category = number2category[box.class_index]
        RGB = category2color[category]
        BGR = (RGB[2],RGB[1],RGB[0])
        #RGB
        position = CoordinateTransform(box.coord,[1280,720]) #list(map(lambda x: int(x), box[0]))
        line_thikness = 4
        cv2.rectangle(image_labeled, (position[0],position[1]), (position[2],position[3]), BGR, line_thikness)
        font = 0
        font_size = 1
        font_thikness = 2
        (text_width, text_height) = cv2.getTextSize(str(category), font, font_size, font_thikness)[0]
        cv2.rectangle(image_labeled, (position[0]- int(line_thikness/2)  ,position[1] - text_height - 3*line_thikness ), (position[0]+ text_width ,position[1] ), BGR, cv2.FILLED)
        cv2.putText(image_labeled, str(category), (position[0],position[1]-10), font, font_size, (255,255,255), 2)
    return image_labeled
    
'''
======================= 施工区 ==========================
'''



#input   : 
#output  : 
#function:    
def Evaluate():
    pass
    
    
'''
======================= 施工区 ==========================
'''

# main workflow:
# for a batch of test set:
#   1. from output of NN to a bbox list(98) :: GetBbox
#   2. use bbox list(98) to create new bbox list with class probability :: Process
#   3. read image, draw bbox and show image :: ReadImage, ShowImage

# 4. evaluate result by mean average precision
def main(nn_output_list,grid_information_dict,threshold,image_name_list,image_path,category2color,number2category):
    for i in range(len(image_name_list)):
        bbox_list = GetBbox(nn_output_list[i], grid_information_dict)
        #TestPrint(bbox_list,"bbox_list")
        bbox_list_processed = Process(bbox_list, threshold)
        image = ReadImage(image_name_list[i],image_path)
        ShowImage(image,image_name_list[i],bbox_list_processed,category2color,number2category)
        
    #Evaluate()#???

    
    
if __name__ == '__main__':

    nn_output_list= [np.load('./test_input/4d44ad18-a4565c5f.npy')]
    #nn_output_list = []
    #TestPrint(nn_output_list[0],"nn_output_list")
    grid_information_dict = {}
    grid_information_dict["cell_row_len"] = 15
    grid_information_dict["cell_column_len"] = 15
    grid_information_dict["bbox_number"] = 2
    grid_information_dict["class_number"] = 10
    
    threshold = 0.01
    image_name_list = ["4d44ad18-a4565c5f.jpg"]
    image_path = "./test//"
    category2color = CCL.CreateColorList()
    number2category = CNL.CreateNumberList(versa = True) 
    main(nn_output_list,grid_information_dict,threshold,image_name_list,image_path,category2color,number2category)
    
    
    
    
    