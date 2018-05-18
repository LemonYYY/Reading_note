## 摘要

这篇论文针对多尺度的目标检测问题，提出了一种能够降低计算量又能保持检测准确率的检测框架。这是一个从粗到细的检测过程，首先在图像的低分辨率版本的的图像中检测，然后在被认为能够提高检测准确率的更高分辨率区域中去fine-tuning。这个pipe-line利用强化学习来实现，包括R-net和Q-net，R-net是用于利用粗检测结果,预测可能提高accuacy的高分辨率区域；Q-net则是在R-net之后用来选择接下来放大的区域。

## 正文

目前大多数显示设备中的图像都比ImageNet和COCO数据集中的分辨率要高很多，如果使用目前主流的CNN-based检测框架，那么处理起来计算量会很大。现在解决这个问题的方法，要么是直接将原图分成几个部分再进入detector，要么直接将原图下采样后再进去detector，这些方法都存
在各自的问题，前者的计算量仍旧巨大，后者的效果不够好。

作者提出的方法如下图所示，通过对图像的下采样版本进行coarse detection，找到需要focus的区域，然后将该区域的高分辨率版本作为fine detector 的输入，达到降低计算量的效果R-net通过学习coarse和fine detection之间的连接来预测放大一个区域的accuracy gain，而 Q-net则是一个用于选择放大（zoom-in）区域的Q-fucntion网络，它是一个DQN，能给出选择某个放大区域的Q-value从而选择一些最好的放大区域。

>感觉这更像是一种更加直接的attention



![image](https://raw.githubusercontent.com/LemonYYY/Reading_note/master/images/Dynamic%20Zoom-in%20Network%20for%20Fast%20Object%20Detection%20in%20Large%20Images/reading_pic_1.jpg)


### Dynamic zoom-in network

本文所提出的检测框架的核心是如何选择那些需要放大的区域，他们用一个Dynamic zoom-in network实现这个操作，整个框架如下图所示：

![image](https://raw.githubusercontent.com/LemonYYY/Reading_note/master/images/Dynamic%20Zoom-in%20Network%20for%20Fast%20Object%20Detection%20in%20Large%20Images/paper_imges2.png)

其中R-net会出输出一个跟输入图片size一样的AG MAP，里面的值代表某个像素点进一步分析会得到的Accuracy gain，某个区域如果进行了进一步的分析，那么该区域所对应的AG MAP里面的值将会降低。

由于使用了强化学习，作者为这个任务设置如下：

- **action** ： 选择将要进一步作高分辨率分析的区域（用bounding boxes 来表示）
- **State** :包括两个信息，1)还未被进一步分析过的区域的accuracy gain，来自R-net；2)被分析过的区域的history，即该区域的accuracy gain的变化轨迹
- **Cost-aware reward function** ： 为了保证在有限的计算量内达到良好的准确率，本文为Q-net设置的reward函数是与计算量相关的。如下图所示，其中 *k in a* 代表k像素值在a这个action选的区域中，`$p^l_k$`代表粗检测的分数，`$p^h_k$`代表细检测的分数，`$g_k$`是ground truth，b代表在action选择区域中的像素个数，B则是整张图的像素个数。
    ![image](https://raw.githubusercontent.com/LemonYYY/Reading_note/master/images/Dynamic%20Zoom-in%20Network%20for%20Fast%20Object%20Detection%20in%20Large%20Images/reading_pic_1_3.png)
    
    第一项代表accuracy improvement，第二项（像素点的比值）则是zoom-in cost





#### R-net

R-net的作用是预测一个与输入图像一样大小的AG map，代表每个像素点放大后的Accuracy Gain，于是他需要学习coarse detection 与 fine detection 之间的关系。

训练的时候需要pre-train的coarse detector 和 fine detector，分别是用低分辨率作为输入和高分辨率图像作为输入,然后设计一个**match layer**来关联这两个detector的输出，当coarse detector的输出 i 与 fine detector的输出 j 的预测框（Bbox）之间IoU大于0.5，则认为该区域是相关的,这个相关的信息用 `$\{(d^l_k , p^l_k , p^h_k , f^l_k)\}$` 来表示，其中d代表检测出来的bounding boxes，p代表置信度，f代表将要放大的feature vector。

>直接将图片的高分辨率版本作为detector的输入，并不一定就能获得很好的结果，例如，当我们detector的训练数据中存在很多小目标时，大的目标就可能更容易在低分辨率的时候检测得更准，因为它占整个输入图像的比例较小，与训练数据中的情况更吻合。

R-net需要利用上面得到的相关性信息`$\{(d^l_k , p^l_k , p^h_k , f^l_k)\}$` 来判断每个区域放大后的accuracy gain。

- 作者定义，当high resolution score `$ p^h_k$` 比 low resolution socre `$p^l_k$`更接近ground truth `$g_k$`的时候，就认为该区域值得放大,否则就不应该放大该区域；

- 文中使用一个相关性回归（Correlation Regression）来评估accuracy gain，如下式所示 

    ![image](https://raw.githubusercontent.com/LemonYYY/Reading_note/master/images/Dynamic%20Zoom-in%20Network%20for%20Fast%20Object%20Detection%20in%20Large%20Images/pic_4.png)
    
    其中，`$\Phi$` 代表回归方程，`$W$`代表参数，其实这里是用一个两个fc层来代替回归方程，输出是就作为accuracy gain，上面的（2）式作为loss，该网络的作用是去逼近前面一项（high resolution 比low resolution 更接近ground truth 的程度，也就是accuracy improvement）。
    
    
- R-net作预测的时候，直接用上面训练好的Regression layer 来预测每个区域的的accuracy gain，AG map 可以通过预测每个proposals所对应区域的accuracy gain 来获得（regression layer 的输出 除以该区域的像素点个数，然后平均分配在区域中的每个像素点中作为AG map 的值）。在放大了某个区域并对该区域的高分辨率版本进行检测后，该区域的accuracy gain 设置为0；


#### Q-net

该部分用于选择将要放大的区域，使用deep q-learning network来实现，这里的公式就如Deepmind那篇文章的一样，监督信息就是 用当前的参数去估计选择某个action的Q值 加上 选择该action的即时reward(式子1定义的)，loss 如下图所示。

![image](https://raw.githubusercontent.com/LemonYYY/Reading_note/master/images/Dynamic%20Zoom-in%20Network%20for%20Fast%20Object%20Detection%20in%20Large%20Images/pic_5.png)


如框架图中所示，Q-net中的卷积神经网络是双支路的，主要是建立不同的感受野，因为输出的map中每一个值都代表该点所对应感受野的Q-value，如图所示，一共有4+9种区域可以选择，在选择的时候，是13个action一起作比较的。



- Window selection refinement
    
  如果直接用Q-net的输出作为放大的窗口，那么可选择的窗口就太少了，作者这里提出了可以对Q-net所作的选择进行调整，调整的方向如下式所示：

    ![image](https://raw.githubusercontent.com/LemonYYY/Reading_note/master/images/Dynamic%20Zoom-in%20Network%20for%20Fast%20Object%20Detection%20in%20Large%20Images/eq_6.png)
    
  其中 `$\hat{a}$` 代表选择重新调整的框，`$A = (x_q \pm \mu_x,y_q \pm \mu_y,w,h)$` 代表调整后的区域，`$A = (x_q, y_q, w, h)$`是Q-net选择的区域，这里的 `$\mu$` 控制了框的调整，我认为是预设的。
  
  作调整的效果图如下图所示：
  
  ![image](https://raw.githubusercontent.com/LemonYYY/Reading_note/master/images/Dynamic%20Zoom-in%20Network%20for%20Fast%20Object%20Detection%20in%20Large%20Images/figure_3.png)



## Experiments

这里主要分析三个定性的对比实验吧（还有一些定量的实验我没有关注）:

1. the effect of the refinement：

    如上图3所示，Q-net的输出所能涵盖的区域太稀疏了，不容易框准将要放大的区域。
    
2. Q-net-CNN + Rnet 与 GS(greedy strategy)+Rnet 的对比：
    
    可以从图4中看到，GS+Rnet 更倾向于找到一些有重叠的放大区域，而Q-net-CNN + Rnet则有时候不会选到最优的放大区域，但是多次迭代后效果会更好。

3. R-net 和 ER(Etropy Region)的对比
    
    这里的ER是指直接检测网络输出的熵来评价coarse detection的质量，高的熵值代表检测效果较低，这样它就没有考虑coarse detection 和 fine detection之间的联系

    如图5所示，在第一列中的detection区域不需要进一步放大，因为效果已经很好了，R-net预测的accuracy gains 很小；在第二列的detection区域中需要进一步放大，R-net的预测值很大；第三列展示了一些在高分辨率情况下效果变差的图片，ER不能判断放大这些区域是否有帮助，但是R-net却能在这种情况下给出负的accuracy gain.

![image](https://raw.githubusercontent.com/LemonYYY/Reading_note/master/images/Dynamic%20Zoom-in%20Network%20for%20Fast%20Object%20Detection%20in%20Large%20Images/figure_4.png)

![image](https://raw.githubusercontent.com/LemonYYY/Reading_note/master/images/Dynamic%20Zoom-in%20Network%20for%20Fast%20Object%20Detection%20in%20Large%20Images/figure_5.png)


 


