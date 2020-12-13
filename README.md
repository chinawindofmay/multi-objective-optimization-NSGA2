# multi-objective-optimization-NSGA2
multi-objective optimization NSGA2
## A_SGA_QGA_master_0610 
## A_SGA_TSP
## A_SGA_with_quantum_0620文件夹
（5）QGA.py 原始的量子遗传算法；
（6）QGA_numpy.py 经Numpy改造的量子遗传算法；
（7）QGA_numpy_elite.py 经Numpy改造，并加入elite机制的量子遗传算法；
（8）QGA_numpy_elite_comprason.py 经Numpy改造，并加入elite机制的量子遗传算法与普通遗传算法的对比；

## B_MOO_MOEAD0709
- 参考代码；
## B_MOO_NSGA2_0710
- 这是晓风提供的代码，根据MoeaPlat的MATLAB代码改写的Python，这个代码存在问题是运行效率慢。
## B_MOO_NSGA2_0817未完成改造
- 无效代码
## B_MOO_NSGA3_0810_PS
- PS-MOOPS-SL求解，行路径规划，初始化等相关的代码；
- 基于老的数据结构的方式进行的计算；
## B_MOO_NSGA3_0920_THEROY
- 一开始是为了证明NSGA3在超目标下是正确的，所以写了这个代码；
- 后来为了简化问题，从nsga2做起；


B_NSGA2_with_qutanum_0630文件夹
（1）NSGA2_01_traditional.py  基于LIST的方式，实现了两个自定义函数(都求最大值)的求解和二维可视化表达（仍存在不足），增加了ndset的测试；
（2）NSGA2_02_math_valiation.py  基于LIST的方式，实现了三个自定义函数（一个求最小值，两个求最大值）的NSGA2求解和三维可视化表达；
（3）NSGA2_math_03_valiation_np.py 基于numpy，实现了三个自定义函数（一个求最小值，两个求最大值）的NSGA2求解和三维可视化表达；
（4）NSGA2_math_03_valiation_np_ZDT3.py 基于numpy，实现了ZDT3函数的NSGA2求解和三维可视化表达；
（5）NSGA2_04_math_valiation_numpy_add_QGA_OLD.py 基于numpy，实现了三个自定义函数（一个求最小值，两个求最大值）的量子NSGA2求解和三维可视化表达，目前集成了量子编码，但是交叉和变异仍是随机的；
（6）NSGA2_



B_MOO_NSGA3_0810_PS文件夹
文件：
（1）nsga3.py
(2)utils.py
起源：
这两个文件夹是从B_MOO_NSGA3_0710中复制的，这块的代码起源于：https://blog.csdn.net/Fengfeng__y/article/details/93776983   哈工大一个同学复现的算法。
修改：
主要将其修改为符合PS选址的功能模块，用于MOO选址
