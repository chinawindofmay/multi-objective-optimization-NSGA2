from  QGA_numpy_elite import *
from simple_ga_2 import *

qga = QGA(POP_SIZE, X_NUM, CHROMOSOME_LENGTH, MAX_VALUE, MIN_VALUE, ITER_NUM, DETA,F3)
best_fitness_results, mean_fitness_results, median_fitness_results, all_record_fitnesses=qga.main()
plot_lines(best_fitness_results, mean_fitness_results, median_fitness_results,"-","QGA",False)

Z_max_result,Z_mean_result,Z_median_result,Z_result=traditional_ga(F3)

plot_lines(Z_max_result,Z_mean_result,Z_median_result,"--","NGA",True)

