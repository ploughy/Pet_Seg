# Pet_Seg
使用Unet对数据集中图片中的猫狗轮廓进行分割
本代码使用的数据集是Oxford-IIIT Pet数据集，官网：https://www.robots.ox.ac.uk/~vgg/data/pets 。
数据集包含两个压缩文件。
原始图像储存在images文件中；
分割图像储存在annotations文件中。
  由于文件中的图像都是随意堆放在文件夹内的，图像的大小、格式以及彩色模式各不相同，还包括了3张图片的mat文件；
所以在进行数据预处理时需要将images文件夹内的图像调整为各项格式相统一的图像。同时将iamges中的图像中的15%分出去作为测试集。


  本项目中的dataset1.py文件即完成了对图像数据的处理。
_sort_images()函数将每张图像的储存路径读入数组中，返回一个经过sorted()函数完成排序之后的数组（其中标签图像要将以‘.’开头的和以‘.mat’结尾的图片文件排除在外）
write_file()函数将被读入数组中的图片的路径按拍好的顺序写入txt文档中，共写入三个文档，分别为用于训练的train.txt,用于测试
的test.txt和用于预测的predict.txt。每个文档中将原图路径和其对应的标签图像路径写在一行，中间用空格分开。
  经过上述处理后可保证项目每次读到的原图都能取到与之相对应的标签图进行训练。
  随后在SEGData()类中，根据mode的赋值（'train','test','predict'）从相应的txt文件中读取图片路径，将图像转换为统一的IMAGE_SIZE，IMAGE_SIZE=（160,160），
然后将图像的色彩模式装化为统一的格式（RGB、RGBA、L或grayscale）

  UNet.py文件完成了对UNet模型的搭建
在搭建UNet模型时做了一些细节的修改，将下采样的max pooling改为了用卷积层替代；将上采样的（插值）改为了转置卷积。

  train.py
对此项目的训练，考虑到本身计算机性能的限制，batch_size设为2，epoch设为15，采用CrossEntropyLoss()计算损失，采用SGD优化器更新权重，学习率设为0.01，
训练模型储存为.pth文件，其中Pet_UNet_ep15_idx3000.pth文件在测试中分割效果最好。

  test.py
读取test.txt文件中的images剩余的图像，加载已训练好的pth文件对图像进行分割。
