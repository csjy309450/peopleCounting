# -*-encoding=utf-8-*-

import os
import cv2
import numpy as np
import time

# image_name_list = []  # read image
# with open(image_path + list_files) as fid:
#     image_name_list = [x.strip() for x in fid.readlines()]
# images_count = len(image_name_list)
#
# data_img = np.zeros((images_count, height, width, 3), np.uint8)
# # data_label = np.zeros((images_count, 2), np.uint8)
# data_label = np.zeros((images_count), np.uint8)
# data_index = np.zeros((images_count), np.uint8)
#
# def loadData():
#     for idx in range(images_count):
#         img = cv2.imread(image_name_list[idx])
#         img = cv2.resize(img, (height, width))
#         data_img[idx, :, :, :] = img
#
#     for idx in range(images_count):
#         if idx < 1201:
#             # data_label[idx] = [0, 1]
#             data_label[idx] = 0
#         else:
#             # data_label[idx] = [1, 0]
#             data_label[idx] = 1
#
# def getBatch_index(batchSize):
#     global data_index
#     data_index = np.random.randint(0, 2202, [batchSize])
#
# def getBatch_img(batchSize, step):
#     # return data[step*batchSize:(step+1):batchSize]
#     start = step*batchSize
#     end = (step+1)*batchSize
#     return data_img[start:end]
#
# def getBatch_label(batchSize, step):
#     # return data[step*batchSize:(step+1):batchSize]
#     start = step * batchSize
#     end = (step + 1) * batchSize
#     return data_label[start:end]
LOAD_STATIC_TO_MEMORY = 1
LOAD_DYNAMIC_TO_MEMORY = 2
class BatchReader:
    def __init__(self, image_path, list_files, reSize, batchSize=None, _loadMode=LOAD_STATIC_TO_MEMORY, _shuffle=False):
        """
        构造函数,完成样本路径及标签的读取
        Arg:
        :_mode:
            LOAD_STATIC_TO_MEMORY: 将所有图片数据架子到内存，只适用于数据集较小的情况
            LOAD_DYNAMIC_TO_MEMORY: 仅加载指定路径的图片入内存，速度较慢，但更灵活
        """
        assert((type(list_files) == tuple and len(list_files) == 2) or type(list_files) == str)

        self.reSize = reSize                # 调整取出图片的大小
        self.batchSize = batchSize          # 每一轮取数据个数
        self.curr_step = 0                  # 当前是第几轮取数据
        self.flag_loadMode = _loadMode      # 指示动态夹数据或静态加载数据
        self.flage_isAllDataInMem = False   # 标记当前是否已经将所有训练数据加载入内存
        self.flag_shuffle = _shuffle        # 是否对数据重新洗牌
        self.image_name_list = []           # 存放图片的存放路径列表
        self.data_label = []                   # 存放图片的标签

        """加载存放文件名换个标签的txt文件"""
        if (type(list_files) == tuple and len(list_files) == 2):
            """图片文件名与标签分开存放的情况"""
            with open(os.path.join(image_path, list_files[0])) as fid:
                self.image_name_list = [x.strip() for x in fid.readlines()]
            with open(os.path.join(image_path, list_files[1])) as fid:
                # self.data_label = [x.strip() for x in fid.readlines()]
                self.data_label = np.array(fid.readlines(), np.uint8)
            assert (len(self.image_name_list) == len(self.data_label))
        elif type(list_files) == str:
            """图片文件名与标签合在一起存放的情况"""
            pass

        self.images_count = len(self.image_name_list)                       # 总的训练数据量
        self.total_step = int(self.images_count / self.batchSize)           # 计算按照给定的batchsize和样本总数，多少满足多少轮取数据操作
        self.data_img = np.empty((0, reSize[1], reSize[0], 3), np.uint8)    # 存放图像的缓冲区
        self.data_index = np.empty((self.batchSize), np.uint8)              # 存放下标
        # 预加载数据,主要完成当选择静态方式时加载数据
        self._preLoad()

    def _preLoad(self):
        """
        预加载数据,主要完成当选择静态方式时加载数据
        :return:
        """
        if self.flag_loadMode == LOAD_STATIC_TO_MEMORY:
            # 采用静态加载时,直接一次性讲数据加载入内存
            # load images
            for idx in range(self.images_count):
                img = cv2.imread(self.image_name_list[idx])
                img = cv2.resize(img, self.reSize)
                # self.data_img[idx, :, :, :] = img
                self.data_img = np.insert(self.data_img, self.data_img.shape[0], img, axis=0)
                self.flag_dataInMem = True
        elif self.flag_loadMode == LOAD_DYNAMIC_TO_MEMORY:
            pass

    def _loadImages(self):
        """
        每一轮调用加载数据
        """
        if self.flag_loadMode == LOAD_STATIC_TO_MEMORY:
            self._setBatch_index()
            return True
        elif self.flag_loadMode == LOAD_DYNAMIC_TO_MEMORY:
            self._setBatch_index()
            self.data_img = np.empty((0, self.reSize[1], self.reSize[0], 3), np.uint8)
            for idx in self.data_index:
                img = cv2.imread(self.image_name_list[idx])
                img = cv2.resize(img, self.reSize)
                # self.data_img[idx, :, :, :] = img
                self.data_img = np.insert(self.data_img, self.data_img.shape[0], img, axis=0)
            return True
        else:
            return False

    def getBatch_index(self):
        """
        读取本轮产生的索引
        :return: 
        """
        return self.data_index

    def _setBatch_index(self):
        """
        产生本轮的索引
        :return: 
        """
        if self.flag_shuffle:
            self.data_index = np.random.randint(0, self.images_count, [self.batchSize])
        else:
            self.data_index = np.arange(self.curr_step*self.batchSize, (self.curr_step+1)*self.batchSize)

    def getBatchData(self):
        """
        用户每轮读取样本的接口
        :return: 
        """
        if self._loadImages():
            self.curr_step += 1
            return True
        return False

    def getBatch_img(self):
        if self.flag_loadMode == LOAD_STATIC_TO_MEMORY:
            return self.data_img[self.data_index]
        elif self.flag_loadMode == LOAD_DYNAMIC_TO_MEMORY:
            return self.data_img
        else:
            pass

    def getBatch_label(self):
        # return data[step*batchSize:(step+1):batchSize]
        # print type(self.data_index)
        return self.data_label[self.data_index]

    def getTestImages(self, step):
        start = self.batchSize*step
        end = self.batchSize*(step+1)
        return self.data_img[start:end]

    def getTestLabels(self, step):
        start = self.batchSize * step
        end = self.batchSize * (step + 1)
        return self.data_label[start:end]

    def getDataCount(self):
        return len(self.image_name_list)

if __name__ == "__main__":
    # loadData()
    # a = getBatch(64, 0)
    # pass

    # getBatch_index(64)
    # print data_index
    image_path = '/home/yangzheng/testData/BodyDataset/'
    list_files = ('train_images.txt', 'train_labels.txt')
    height = 24
    width = 24
    # bacthSize = 64
    batchReader = BatchReader(image_path, list_files, (width, height), 64, _shuffle=True)
    batchReader._loadImages()
    print batchReader.getBatch_label()
    print batchReader.getBatch_index()
    batchReader._loadImages()
    print batchReader.getBatch_label()
    print batchReader.getBatch_index()
    batchReader._loadImages()
    print batchReader.getBatch_label()
    print batchReader.getBatch_index()
    batchReader._loadImages()
    print batchReader.getBatch_label()
    print batchReader.getBatch_index()
    # print batchReader.getBatch_index()
    # time.sleep(5)
    # _ = batchReader.getBatchData()
    # print batchReader.getBatch_index()
    # time.sleep(5)
    # _ = batchReader.getBatchData()
    # print batchReader.getBatch_index()
    # time.sleep(5)
    # _ = batchReader.getBatchData()
    # print batchReader.getBatch_index()
    # a.getBatch_index()
    # print a.data_index
    # print a.data_label[1500]
    # print a.getBatch_label()
    pass