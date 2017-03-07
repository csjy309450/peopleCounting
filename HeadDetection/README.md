# peopleCounting
Moduls in people counting system

算法的思路：
1.使用类似方向梯度的方法提取视频每一帧的边缘信息
2.是用Vibe算法做背景分割
3.使用haar-like特征+cascade分类器探测图像中所有的人头
