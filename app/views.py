from django.shortcuts import render

# Create your views here.
import numpy as np #矩阵运算
import urllib #url解析
import json #json字符串使用
import cv2 #openCV2包
import os # 执行操作系统命令
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
