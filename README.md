# multi-objective-optimization-NSGA2
multi-objective optimization NSGA2

这是一个测试代码；
（1）NSGA 2-math-valiation.py  实现了三个函数（一个求最小值，两个求最大值）的NSGA2求解和三维可视化表达；
（2）NSGA 2-traditional.py  实现了两个函数的求解和二维可视化表达（仍存在不足）；
（3）NSGA 2-math-valiation_numpy.py 实现了 numpy机制下的三个函数（一个求最小值，两个求最大值）的NSGA2求解和三维可视化表达；
（4）NSGA 2-math-valiation_numpy_add_QGA.py 实现了 numpy机制下的三个函数（一个求最小值，两个求最大值）的量子NSGA2求解和三维可视化表达，目前集成了量子编码，但是交叉和变异仍是随机的；
（5）QGA.py 原始的量子遗传算法；
（6）QGA_numpy.py 经Numpy改造的量子遗传算法；
（7）QGA_numpy_elite.py 经Numpy改造，并加入elite机制的量子遗传算法；
（8）QGA_numpy_elite_comprason.py 经Numpy改造，并加入elite机制的量子遗传算法与普通遗传算法的对比；