# multi-objective-optimization-NSGA2
multi-objective optimization NSGA2

A_SGA_with_quantum_0620文件夹
（5）QGA.py 原始的量子遗传算法；
（6）QGA_numpy.py 经Numpy改造的量子遗传算法；
（7）QGA_numpy_elite.py 经Numpy改造，并加入elite机制的量子遗传算法；
（8）QGA_numpy_elite_comprason.py 经Numpy改造，并加入elite机制的量子遗传算法与普通遗传算法的对比；

NSGA2_with_qutanum_0630文件夹
（1）NSGA2_01_traditional.py  基于LIST的方式，实现了两个自定义函数(都求最大值)的求解和二维可视化表达（仍存在不足），增加了ndset的测试；
（2）NSGA2_02_math_valiation.py  基于LIST的方式，实现了三个自定义函数（一个求最小值，两个求最大值）的NSGA2求解和三维可视化表达；
（3）NSGA2_math_03_valiation_np.py 基于numpy，实现了三个自定义函数（一个求最小值，两个求最大值）的NSGA2求解和三维可视化表达；
（4）NSGA2_math_03_valiation_np_ZDT3.py 基于numpy，实现了ZDT3函数的NSGA2求解和三维可视化表达；
（5）NSGA2_04_math_valiation_numpy_add_QGA_OLD.py 基于numpy，实现了三个自定义函数（一个求最小值，两个求最大值）的量子NSGA2求解和三维可视化表达，目前集成了量子编码，但是交叉和变异仍是随机的；
（6）NSGA2_