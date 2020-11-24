# ImageDetection
Author dcyao
## 利用Python Django 搭建的服务后台
### 需要的python 库有opencv 4.0 和常用的python库 
启动脚本为
```bash
python manage.py runserver 8000 
```
---
函数名称|函数功能|已经实现
--|:--:|--:
code_001|图片读取与显示|✔
code_002|	图片灰度化	|✔️
code_003|	图像创建与赋值	✔️
code_004	图像像素读写	✔️
code_005	图像像素算术操作(加减乘除)	✔️
code_006	图像伪彩色增强	✔️
code_007	图像像素操作(逻辑操作)	✔️
code_008	图像通道分离合并	✔️
code_009	色彩空间与色彩空间转换	✏️
code_010	图像像素值统计	✔️
code_011	图像像素归一化	✔️
code_012	视频读写	✔️
code_013	图像翻转	✔️
code_014	图像插值	✔️
code_015	绘制几何形状	✔️
code_016	图像ROI与ROI操作	✔️
code_017	图像直方图	✔️
code_018	图像直方图均衡化	✏️
code_019	图像直方图比较	✔️
code_020	图像直方图反向投影	✔️
code_021	图像卷积操作	✔️
code_022	图像均值与高斯模糊	❣️
code_023	中值模糊	✔️
code_024	图像噪声	✔️
code_025	图像去噪声	✔️
code_026	高斯双边模糊	✔️
code_027	均值迁移模糊(mean-shift blur)	✔️
code_028	图像积分图算法	✔️
code_029	快速的图像边缘滤波算法	✔️
code_030	自定义滤波器	✔️
code_031	Sobel算子	✔️
code_032	更多梯度算子	✔️
code_033	拉普拉斯算子(二阶导数算子)	✔️
code_034	图像锐化	✔️
code_035	USM 锐化增强算法	✔️
code_036	Canny边缘检测器	❣️
code_037	图像金字塔	✔️
code_038	拉普拉斯金字塔	✔️
code_039	图像模板匹配	✔️
code_040	二值图像介绍	✔️
code_041	基本阈值操作	✔️
code_042	图像二值寻找法OTSU	✏️
code_043	图像二值寻找法TRIANGLE	✔️
code_044	图像自适应阈值算法	✏️
code_045	图像二值与去噪	✏️
code_046	图像连通组件寻找	✔️
code_047	图像连通组件状态统计	✔️
code_048	轮廓寻找	❣️
code_049	轮廓外接矩形	❣️
code_050	轮廓矩形面积与弧长	✏️
code_051	轮廓逼近	✔️
code_052	几何矩计算中心	✔️
code_053	使用Hu矩阵实现轮廓匹配	✔️
code_054	轮廓圆与椭圆拟合	✔️
code_055	凸包检测	✏️
code_056	直线拟合与极值点寻找	✔️
code_057	点多边形测试	✔️
code_058	寻找最大内接圆	✔️
code_059	霍夫曼直线检测	✔️
code_060	概率霍夫曼直线检测	❣️
code_061	霍夫曼圆检测	❣️
code_062	膨胀和腐蚀	❣️
code_063	结构元素	✔️
code_064	开运算	✏️
code_065	闭运算	✏️
code_066	开闭运算的应用	✏️
code_067	顶帽	✔️
code_068	黑帽	✔️
code_069	图像梯度	✔️
code_070	基于梯度的轮廓发现	✏️
code_071	击中击不中	✔️
code_072	缺陷检测1	✔️
code_073	缺陷检测2	✔️
code_074	提取最大轮廓和编码关键点	✔️
code_075	图像修复	✔️
code_076	图像透视变换应用	✏️
code_077	视频读写和处理	✏️
code_078	识别与跟踪视频中的特定颜色对象	✔️
code_079	视频分析-背景/前景 提取	✔️
code_080	视频分析–背景消除与前景ROI提取	✔️
code_081	角点检测-Harris角点检测	✔️
code_082	角点检测-Shi-Tomas角点检测	✏️
code_083	角点检测-亚像素角点检测	✔️
code_084	视频分析-KLT光流跟踪算法-1	✏️
code_085	视频分析-KLT光流跟踪算法-2	✏️
code_086	视频分析-稠密光流分析	✏️
code_087	视频分析-帧差移动对象分析	✔️
code_088	视频分析-均值迁移	✏️
code_089	视频分析-连续自适应均值迁移	✏️
code_090	视频分析-对象移动轨迹绘制	✔️
code_091	对象检测-HAAR级联分类器	✔️
code_092	对象检测-HAAR特征分析	✔️
code_093	对象检测-LBP特征分析	✔️
code_094	ORB 特征关键点检测	✏️
code_095	ORB 特征描述子匹配	✔️
code_096	多种描述子匹配方法	✏️
code_097	基于描述子匹配的已知对象定位	✏️
code_098	SIFT 特征关键点检测	✔️
code_099	SIFT 特征描述子匹配	✔️
code_100	HOG 行人检测	✔️
code_101	HOG 多尺度检测	✏️
code_102	HOG 提取描述子	✔️
code_103	HOG 使用描述子生成样本数据	✔️
code_104	(检测案例)-HOG+SVM 训练	✔️
code_105	(检测案例)-HOG+SVM 预测	✔️
code_106	AKAZE 特征与描述子	✔️
code_107	Brisk 特征与描述子	✔️
code_108	GFTT关键点检测	✔️
code_109	BLOB 特征分析	✔️
code_110	KMeans 数据分类	✔️
code_111	KMeans 图像分割	✔️
code_112	KMeans 图像替换	✔️
code_113	KMeans 图像色卡提取	✔️
code_114	KNN 分类模型	✔️
code_115	KNN 数据保存	✔️
code_116	决策树算法	✔️
code_117	图像均值漂移分割	✔️
code_118	Grabcut-图像分割	✔️
code_119	Grabcut-背景替换	✏️
code_120	二维码检测识别	✔️
code_121	DNN- 读取模型各层信息	✔️
code_122	DNN- DNN实现图像分类	✔️
code_123	DNN- 模型运行设置目标设备与计算后台	✔️
code_124	DNN- SSD单张图片检测	✔️
code_125	DNN- SSD实时视频检测	✔️
code_126	DNN- 基于残差网络的人脸检测	✔️
code_127	DNN- 基于残差网络的视频人脸检测	✔️
code_128	DNN- 调用tensorflow的检测模型	✔️
code_129	DNN- 调用openpose模型实现姿态评估	✔️
code_130	DNN- 调用YOLO对象检测网络	✔️
code_131	DNN- YOLOv3-tiny版本实时对象检测	✔️
code_132	DNN- 单张与多张图像的推断	✔️
code_133	DNN- 图像颜色化模型使用	✔️
code_134	DNN- ENet实现图像分割	✔️
code_135	DNN- 实时快速的图像风格迁移
