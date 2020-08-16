#########################################################
#                                                       #
#       QUANTUM GENETIC ALGORITHM (24.05.2016)          #
#                                                       #
#               R. Lahoz-Beltra                         #
#                                                       #
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND   #
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY #
# AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  #
# THE SOFWTARE CAN BE USED BY ANYONE SOLELY FOR THE     #
# PURPOSES OF EDUCATION AND RESEARCH.                   #
#                                                       #
#########################################################
import math
import numpy as np
import matplotlib.pyplot as plt

#########################################################
# ALGORITHM PARAMETERS                                  #
#########################################################
N=200                  # Define here the population size
Genome=10              # Define here the chromosome length
generation_max=500    # Define here the maximum number of
                      # generations/iterations

#########################################################
# VARIABLES ALGORITHM                                   #
#########################################################
popSize=N+1
genomeLength=Genome+1
top_bottom=3
QuBitZero = np.array([[1],[0]])
QuBitOne = np.array([[0],[1]])
AlphaBeta = np.empty([top_bottom])
fitness = np.empty([popSize])
probability = np.empty([popSize])
# qpv: quantum chromosome (or population vector, QPV)
qpv = np.empty([popSize, genomeLength, top_bottom])         
nqpv = np.empty([popSize, genomeLength, top_bottom])
# chromosome: classical chromosome
chromosome = np.empty([popSize, genomeLength],dtype=np.int) 
child1 = np.empty([popSize, genomeLength, top_bottom])
child2 = np.empty([popSize, genomeLength, top_bottom])
best_chrom = np.empty([generation_max])

# Initialization global variables
theta=0;
iteration=0;
the_best_chrom=0;
generation=0;

#########################################################
# QUANTUM POPULATION INITIALIZATION                     #
#########################################################
def Init_population():
    # Hadamard gate
    r2=math.sqrt(2.0)
    h=np.array([[1/r2,1/r2],[1/r2,-1/r2]])
    # Rotation Q-gate
    theta=0;
    rot =np.empty([2,2])
    # Initial population array (individual x chromosome)
    i=1; j=1;
    for i in range(1,popSize):
     for j in range(1,genomeLength):
        theta=np.random.uniform(0,1)*90
        theta=math.radians(theta)
        rot[0,0]=math.cos(theta); rot[0,1]=-math.sin(theta);
        rot[1,0]=math.sin(theta); rot[1,1]=math.cos(theta);
        AlphaBeta[0]=rot[0,0]*(h[0][0]*QuBitZero[0])+rot[0,1]*(h[0][1]*QuBitZero[1])
        AlphaBeta[1]=rot[1,0]*(h[1][0]*QuBitZero[0])+rot[1,1]*(h[1][1]*QuBitZero[1])
        # alpha squared          
        qpv[i,j,0]=np.around(2*pow(AlphaBeta[0],2),2) 
        # beta squared
        qpv[i,j,1]=np.around(2*pow(AlphaBeta[1],2),2) 

#########################################################
# SHOW QUANTUM POPULATION                               #
#########################################################
def Show_population():
    i=1; j=1;
    for i in range(1,popSize):
        print()
        print()
        print("qpv = ",i," : ")
        print()
        for j in range(1,genomeLength):
          print(qpv[i, j, 0],end="")
          print(" ",end="")
        print()
        for j in range(1,genomeLength):
          print(qpv[i, j, 1],end="")
          print(" ",end="")
    print()
    
#########################################################
# MAKE A MEASURE                                        #
#########################################################
# p_alpha: probability of finding qubit in alpha state
def Measure(p_alpha):
    for i in range(1,popSize):
        print()
        for j in range(1,genomeLength):
            if p_alpha<=qpv[i, j, 0]:
                chromosome[i,j]=0
            else:
                chromosome[i,j]=1
            print(chromosome[i,j]," ",end="")
        print()
    print()

#########################################################
# FITNESS EVALUATION                                    #
#########################################################
def Fitness_evaluation(generation):
    i=1; j=1; fitness_sum=0; sum_sqr=0;
    fitness_average=0
    variance=0
    for i in range(1,popSize):
        fitness[i]=0
    #########################################################
    # Define your problem in this section. For instance:    #
    #                                                       #
    # Let f(x)=abs(x-5)/(2+sin(x)) be a function that takes   #
    # values in the range 0<=x<=15. Within this range f(x)  #
    # has a maximum value at x=11 (binary is equal to 1011) #
    #########################################################
    for i in range(1,popSize):
       x=0;
       for j in range(1,genomeLength):
           # translate from binary to decimal value
           x=x+chromosome[i,j]*pow(2,genomeLength-j-1)
           # replaces the value of x in the function f(x)
           y= np.fabs((x-5)/(2+np.sin(x)))
           # the fitness value is calculated below:
           # (Note that in this example is multiplied
           # by a scale value, e.g. 100)
           fitness[i]=y*100
       #########################################################
       # print("fitness = ",i," ",fitness[i])
       fitness_sum=fitness_sum+fitness[i]
    fitness_average=fitness_sum/N
    i=1;
    while i<=N:
        sum_sqr=sum_sqr+pow(fitness[i]-fitness_average,2)
        i=i+1
    variance=sum_sqr/N
    if variance<=1.0e-4:
        variance=0.0
    # Best chromosome selection
    the_best_chrom=0;
    fitness_max=fitness[1];
    for i in range(1,popSize):
        if fitness[i]>=fitness_max:
            fitness_max=fitness[i]
            the_best_chrom=i
    best_chrom[generation]=the_best_chrom
    # Statistical output
    f = open("output.dat","a")
    f.write(str(generation)+" "+str(fitness_average)+"\n")
    f.write(" \n")
    f.close()
    f = open("output_all.dat", "a")
    f.write(str(generation) + "#" + ",".join(str(kk) for kk in list(fitness)))
    f.write("&")
    f.close()
    # print("Population size = ",popSize - 1)
    print("mean fitness = ",fitness_average)
    print("variance = ",variance," Std. deviation = ",math.sqrt(variance))
    print("fitness max = ",best_chrom[generation])
    print("fitness sum = ",fitness_sum)

#########################################################
# QUANTUM ROTATION GATE                                 #
#########################################################
def rotation():
    rot=np.empty([2,2])
    # Lookup table of the rotation angle
    for i in range(1,popSize):
       for j in range(1,genomeLength):
           if fitness[i]<fitness[int(best_chrom[generation])]:
             # if chromosome[i,j]==0 and chromosome[best_chrom[generation],j]==0:
               if chromosome[i,j]==0 and chromosome[int(best_chrom[generation]),j]==1:
                   # Define the rotation angle: delta_theta (e.g. 0.0785398163)
                   delta_theta=0.0785398163
                   rot[0,0]=math.cos(delta_theta); rot[0,1]=-math.sin(delta_theta);
                   rot[1,0]=math.sin(delta_theta); rot[1,1]=math.cos(delta_theta);
                   nqpv[i,j,0]=(rot[0,0]*qpv[i,j,0])+(rot[0,1]*qpv[i,j,1])
                   nqpv[i,j,1]=(rot[1,0]*qpv[i,j,0])+(rot[1,1]*qpv[i,j,1])
                   qpv[i,j,0]=round(nqpv[i,j,0],2)
                   qpv[i,j,1]=round(1-nqpv[i,j,0],2)
               if chromosome[i,j]==1 and chromosome[int(best_chrom[generation]),j]==0:
                   # Define the rotation angle: delta_theta (e.g. -0.0785398163)
                   delta_theta=-0.0785398163
                   rot[0,0]=math.cos(delta_theta); rot[0,1]=-math.sin(delta_theta);
                   rot[1,0]=math.sin(delta_theta); rot[1,1]=math.cos(delta_theta);
                   nqpv[i,j,0]=(rot[0,0]*qpv[i,j,0])+(rot[0,1]*qpv[i,j,1])
                   nqpv[i,j,1]=(rot[1,0]*qpv[i,j,0])+(rot[1,1]*qpv[i,j,1])
                   qpv[i,j,0]=round(nqpv[i,j,0],2)
                   qpv[i,j,1]=round(1-nqpv[i,j,0],2)
             # if chromosome[i,j]==1 and chromosome[best_chrom[generation],j]==1:

#########################################################
# X-PAULI QUANTUM MUTATION GATE                         #
#########################################################
# pop_mutation_rate: mutation rate in the population
# mutation_rate: probability of a mutation of a bit
def mutation(pop_mutation_rate, mutation_rate):
    for i in range(1,popSize):
        up=np.random.random_integers(100)
        up=up/100
        if up<=pop_mutation_rate:
            for j in range(1,genomeLength):
                um=np.random.random_integers(100)
                um=um/100
                if um<=mutation_rate:
                    nqpv[i,j,0]=qpv[i,j,1]
                    nqpv[i,j,1]=qpv[i,j,0]
                else:
                    nqpv[i,j,0]=qpv[i,j,0]
                    nqpv[i,j,1]=qpv[i,j,1]
        else:
            for j in range(1,genomeLength):
                nqpv[i,j,0]=qpv[i,j,0]
                nqpv[i,j,1]=qpv[i,j,1]
    for i in range(1,popSize):
        for j in range(1,genomeLength):
            qpv[i,j,0]=nqpv[i,j,0]
            qpv[i,j,1]=nqpv[i,j,1]

#########################################################
# PERFORMANCE GRAPH                                     #
#########################################################
# Read the Docs in http://matplotlib.org/1.4.1/index.html
def plot_Output():
    data = np.loadtxt('output.dat')
    # plot the first column as x, and second column as y
    x=data[:,0]
    y=data[:,1]
    plt.plot(x,y,"r-")
    plt.xlabel('Generation')
    plt.ylabel('Fitness average')
    plt.xlim(0.0, generation_max)
    # plt.show()


#########################################################
# PERFORMANCE GRAPH All RESULT                          #
#########################################################
# Read the Docs in http://matplotlib.org/1.4.1/index.html
def plot_all_output():
    index_list=[]
    data_list=[]
    file = open('output_all.dat', 'r')
    try:
        text = file.read()
        text_line_list=text.split("&")
        for item in text_line_list[0:len(text_line_list)-1]:
            i = int(item.split("#")[0])
            list_obj = np.array(eval(item.split("#")[1]))
            index_list.append(i)
            data_list.append(list_obj)

            # data_list.append([[i,val] for val in list_obj])
    finally:
        file.close()
    plt.plot(data_list, 'g.')
    plt.xlabel('Generation')
    # plt.ylabel('Fitness all points')
    plt.xlim(0, generation_max)
    # plt.yticks(range(0,250,25))
    # plt.ylim(0, 250)
    plt.show()

########################################################
#                                                      #
# MAIN PROGRAM                                         #
#                                                      #
########################################################
def Q_GA():
    generation=0;
    print("============== GENERATION: ",generation," =========================== ")
    print()
    Init_population()
    # Show_population()
    Measure(0.5)
    Fitness_evaluation(generation)
    while (generation<generation_max-1):
      print("The best of generation [",generation,"] ", best_chrom[generation])
      print()
      print("============== GENERATION: ",generation+1," =========================== ")
      print()
      rotation()
      mutation(0.01,0.001)
      generation=generation+1
      Measure(0.5)
      Fitness_evaluation(generation)

print ("""QUANTUM GENETIC ALGORITHM""")
# input("Press Enter to continue...")
Q_GA()
plot_Output()
plot_all_output()












