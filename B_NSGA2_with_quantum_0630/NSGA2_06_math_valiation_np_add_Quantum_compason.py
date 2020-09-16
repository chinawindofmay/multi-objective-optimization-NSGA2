#encoding:utf-8
import NSGA2_03_math_valiation_np as Tradition
import NSGA2_05_evaluation as Evaluation
import NSGA2_05_math_valiation_np_add_Quantum_evaluation_rotate as Rotate
import numpy as np

from B_NSGA2_with_quantum_0630 import NSGA2_05_math_valiation_np_add_Quantum_evaluation_no_rotate as No_rotate

if __name__=="__main__":
    # Main program starts here
    POP_SIZE = 100
    MAX_GEN = 200
    X_COUNT = 2
    CROSSOVER_PROB__THRESHOLD = 0.5
    MUTATION_PROB__THRESHOLD = 0.5
    # Initialization
    MIN_X = 0
    MAX_X = 10
    DELATE = 2e-7
    ANGLE_DETA = 0.05 * np.pi
    DISTANCE_INFINTE = 44444444444444
    CHROMOSOME_LEN=16
    BEGIN_C_M=8

    traditional_nsga2 = Tradition.Traditional_NSGA2(POP_SIZE,
                                            MAX_GEN,
                                            X_COUNT,
                                            CROSSOVER_PROB__THRESHOLD,
                                            MUTATION_PROB__THRESHOLD,
                                            MIN_X,
                                            MAX_X,
                                            DELATE,
                                            DISTANCE_INFINTE,
                                            Evaluation.y1,
                                            Evaluation.y2,
                                            Evaluation.y3)
    traditional_pof_population_np, traditional_pof_y1_values_np, traditional_pof_y2_values_np, traditional_pof_y3_values_np, traditional_gd_array, traditional_sp_array = traditional_nsga2.execute_nsga2()

    no_rotate_nsga2 = No_rotate.Quantum_NSGA2(POP_SIZE,
                              MAX_GEN,
                              X_COUNT,
                              CROSSOVER_PROB__THRESHOLD,
                              MUTATION_PROB__THRESHOLD,
                              MIN_X,
                              MAX_X,
                              DELATE,
                              DISTANCE_INFINTE,
                              Evaluation.y1,
                              Evaluation.y2,
                              Evaluation.y3,
                              CHROMOSOME_LEN,
                              BEGIN_C_M,
                              ANGLE_DETA)
    no_rotate_pof_population_np, no_rotate_pof_y1_values_np, no_rotate_pof_y2_values_np, no_rotate_pof_y3_values_np, no_rotate_gd_array, no_rotate_sp_array = no_rotate_nsga2.execute_quantum_nsga2()  # 执行

    rotate_nsga2=Rotate.Quantum_NSGA2_Rotate(POP_SIZE,
                                           MAX_GEN,
                                           X_COUNT,
                                           CROSSOVER_PROB__THRESHOLD,
                                           MUTATION_PROB__THRESHOLD,
                                           MIN_X,
                                           MAX_X,
                                           DELATE,
                                           DISTANCE_INFINTE,
                                           Evaluation.y1,
                                           Evaluation.y2,
                                           Evaluation.y3,
                                           CHROMOSOME_LEN,
                                           BEGIN_C_M,
                                           ANGLE_DETA)
    rotate_pof_population_np, rotate_pof_y1_values_np, rotate_pof_y2_values_np, rotate_pof_y3_values_np,rotate_gd_array,rotate_sp_array=rotate_nsga2.execute_quantum_nsga2()  #执行
    
    #将GD、SP结果展示出来
    Evaluation.draw_2d_plot_gd_and_sp_compason(MAX_GEN,
                                                traditional_gd_array, traditional_sp_array,
                                                no_rotate_gd_array, no_rotate_sp_array,
                                                rotate_gd_array, rotate_sp_array)
    #将POF展示出来
    Evaluation.draw_2d_plot_evaluation_pof_compason(traditional_pof_y1_values_np, traditional_pof_y2_values_np, traditional_pof_y3_values_np,
                                                    no_rotate_pof_y1_values_np, no_rotate_pof_y2_values_np,no_rotate_pof_y3_values_np,
                                                    rotate_pof_y1_values_np, rotate_pof_y2_values_np,rotate_pof_y3_values_np)
    #将求解函数展示出来
    Evaluation.draw_3d_plot(MIN_X, MAX_X,traditional_pof_population_np,
                            traditional_pof_y1_values_np,
                            traditional_pof_y2_values_np,
                            traditional_pof_y3_values_np,
                            Evaluation.y1,
                            Evaluation.y2,
                            Evaluation.y3)
    # draw_3d_plot_test3333()
