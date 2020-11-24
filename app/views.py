from django.shortcuts import render

# Create your views here.
import numpy as np #矩阵运算
import urllib #url解析
import json #json字符串使用
import cv2 #openCV2包
import chardet
import os # 执行操作系统命令
from operator import methodcaller
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse,HttpResponse
# 图片处理模块
import base64
import json
# 主要是支撑算法模块
class ImgModule():
    # ImgModule 模块初始化
    def __init__(self, img_base64_data):
        # 存储原数据
        self.img_base64_data = self.base64_to_image(img_base64_data)
    # 将openCv2 的图片 转换为对应的base64的数据
    def image_to_base64(self,image_np):
        image = cv2.imencode('.jpg', image_np)[1]
        image_code = str(base64.b64encode(image))[2: -1]
        return image_code
    # 将img的数据变成base64 传递到前端来显示
    def base64_to_image(self,b_base64_code):
        # python3 需要将bytes 转换为str 否则不行的
        # base64_code = str(b_base64_code, encoding='utf-8')
        # format, imgstr = base64_code.split(';base64,')
        format, imgstr = b_base64_code.split(';base64,')
        ext = format.split('/')[-1]
        print('ext', ext)
        # base64解码
        img_data = base64.b64decode(imgstr)
        # 转换为np数组
        img_array = np.fromstring(img_data, np.uint8)
        # 转换成opencv可用格式
        print('img_array', img_array)
        img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
        # 如果需要将这个图像保存 那么可以需要这么做即可
        # cv2.imwrite('bird.jpg', img)
        return img
    # 以下都是为算法
    ###############################################################################
    ##################### 以下算法为图像识别的基础算法 ###########################################
    ##############################################################################
    # canny 边缘检测
    def cannyFilters(self):
        src = cv2.cvtColor(self.img_base64_data, cv2.COLOR_RGB2BGR)

        # t1 = 100, t2 = 3*t1 = 300
        edge = cv2.Canny(src, 100, 300)

        edge_src = cv2.bitwise_and(src, src, mask=edge)

        h, w = src.shape[:2]
        result = np.zeros([h, w * 2, 3], dtype=src.dtype)
        result[0:h, 0:w, :] = src
        result[0:h, w:2 * w, :] = edge_src
        cv2.putText(result, "original image", (10, 30), cv2.FONT_ITALIC, 1.0, (0, 0, 255), 2)
        cv2.putText(result, "edge image", (w + 10, 30), cv2.FONT_ITALIC, 1.0, (0, 0, 255), 2)
        return result
    #Sobel 算子
    def sobelFilters(self):
        src = cv2.cvtColor(self.img_base64_data, cv2.COLOR_RGB2BGR)

        h, w = src.shape[:2]
        x_grad = cv2.Sobel(src, cv2.CV_32F, 1, 0)
        y_grad = cv2.Sobel(src, cv2.CV_32F, 0, 1)

        x_grad = cv2.convertScaleAbs(x_grad)
        y_grad = cv2.convertScaleAbs(y_grad)
        # cv.imshow("x_grad", x_grad)
        # cv.imshow("y_grad", y_grad)

        dst = cv2.add(x_grad, y_grad, dtype=cv.CV_16S)
        dst = cv2.convertScaleAbs(dst)
        return dst
    #Y Flip
    def yFlip(self):
        image_to_read = cv2.flip(self.img_base64_data, 1)
        return image_to_read
    #X Flip
    def xFlip(self):
        image_to_read = cv2.flip(self.img_base64_data, 0)
        return image_to_read
    # 图像像素值统计
    def pixelValueStatistics(self):
        image_to_read = cv2.cvtColor(self.img_base64_data, cv2.COLOR_BGR2GRAY)
        min, max, minLoc, maxLoc = cv2.minMaxLoc(image_to_read)
        means, stddev = cv2.meanStdDev(image_to_read)
        image_to_read[np.where(image_to_read < means)] = 0
        image_to_read[np.where(image_to_read > means)] = 255
        return image_to_read
    # 图片变成灰度图像
    def toGray(self):
        image_to_read = cv2.cvtColor(self.img_base64_data, cv2.COLOR_BGR2GRAY)
        return image_to_read
    # 图像识别
    def detectImage(self):
        # 读取的模型数据
        bin_model = "google_model/google/bvlc_googlenet.caffemodel"
        # 识别的文字信息
        protxt = "google_model/google/bvlc_googlenet.prototxt"
        # Load names of classes
        classes = None
        with open("google_model/google/classification_classes_ILSVRC2012.txt", 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')

        # load CNN model
        net = cv2.dnn.readNetFromCaffe(protxt, bin_model)

        # read input data
        image = self.img_base64_data
        blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (104, 117, 123), False, crop=False)
        result = np.copy(image)
        # Run a model
        net.setInput(blob)
        out = net.forward()
        # Get a class with a highest score.
        out = out.flatten()
        classId = np.argmax(out)
        confidence = out[classId]

        # Put efficiency information.
        t, _ = net.getPerfProfile()
        label = 'cost time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(result, label, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Print predicted class.
        label = '%s: %.4f' % (classes[classId] if classes else 'Class #%d' % classId, confidence)
        print('label', label)
        cv2.putText(result, label, (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return result
def hello(request):
    return HttpResponse("hello, world")
    # return JsonResponse({
    #     "hello": "hello"
    # })
@csrf_exempt
def process_image(request):
    default = {"safely executed": False}
    # 获取图像的编辑函数function name
    # 先要序列化数据
    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)
    funName = body['funName']
    # 获取图像的base64数据
    base64_img = body['img']
    # 引入ImgModule 模块来处理数据
    imgModule = ImgModule(base64_img)
    # 通过映射来执行对应的方法 这个回头可以放在Module 里面

    image_to_read = methodcaller(str(funName))(imgModule)
    # 图片编码
    img64 = imgModule.image_to_base64(image_to_read)
    default["img64"] = img64
    return JsonResponse(default)