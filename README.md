# STCKA-
代码复现 Deep Short Text Classification with Knowledge Powered Attention
说明：测试的代码还没有写完，在获取数据的的时候获取到了str，而需要int，应该是数据集加载那个地方缺少从词转token下标的步骤。训练集上没有问题，文件中的效果是在训练集上的。
只跑了数据集1
# 基础环境：
pytorch1.10
需要cuda。
# 文件说明
一共6个python文件，有的是前期处理数据用的，不需要运行了，我已经处理好放在dataset了。
# run
装完所有需要的包，完成后：
执行
python train.py

![image](https://user-images.githubusercontent.com/49407391/204988581-04eb1d10-955d-4a4b-ae18-e615ec963dc7.png)
