%----------------------------------------------------------------------
%程序功能：实现MOEAD算法，测试函数为ZDT1,ZDT2,ZDT3,ZDT4,ZDT6,DTLZ1,DTLZ2
%说明：遗传算子为模拟二进制交叉和多项式变异
%作者：(晓风)
%email: 18821709267@163.com 
%最初建立时间：2018.09.30
%最近修改时间：2018.10.08
%参考论文：
%MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition
%Qingfu Zhang, Senior Member, IEEE, and Hui Li
%IEEE TRANSACTIONS O
%----------------------------------------------------------
clear all
clc
tic;
%------------------------参数输入--------------------------
format long
global x_max x_min x_num f_num lamda z
rand('state',sum(100*clock));
N=300;%种群大小
T=20;%邻居规模大小
fun='DTLZ2';%
funfun;%测试函数
lamda=genrate_lamda(N,f_num);%均匀分布的N个权重向量
max_gen=250;%进化代数
pc=1;%交叉概率
pm=1/x_num;%变异概率
yita1=2;%模拟二进制交叉参数2
yita2=5;%多项式变异参数5
%------------------------初始条件--------------------------
%%计算任意两个权重向量间的欧式距离，查找每个权向量最近的T个权重向量的索引
B=look_neighbor(lamda,T);
%%在可行空间均匀随机产生初始种群
X=initialize(N,f_num,x_num,x_min,x_max,fun);
%%初始化z
for i=1:f_num
    z(i) = min(X(:,x_num+i));
end
%%初始化是否为非支配个体
X=deterdomination(X,N,f_num,x_num);
%%设置EP为初始种群里的非支配个体
EP=[];
for i=1:N
    if(X(i,x_num+f_num+1)==1)
        EP=[X(i,:);EP];
    end
end
%------------------------迭代更新--------------------------
for gen=1:max_gen
    for i=1:N
        %%基因重组，从B(i)中随机选取两个序列k，l
        index1 = randperm(T);
        parent1 = B(i,index1(1));
        parent2 = B(i,index1(2));
        off=cross_mutation(X(parent1,:),X(parent2,:),f_num,x_num,x_min,x_max,pc,pm,yita1,yita2,fun );
        %off=cross_mutation2(X(parent1,:),X(parent2,:),f_num,x_num,x_min,x_max,pc,pm,yita1,yita2,fun );
        %%更新z
        for j=1:f_num
            %%if(Zi<fi(y')),zi=fi(y')
            z(j)=min(z(j),off(:,x_num+j));
        end
        %%更新领域解
        X=updateNeighbor(lamda,z,X,B(i,:),off,x_num,f_num);
        %%更新EP
        [number,~]=size(EP);
        temp=0;
        kk=[];
        for k=1:number
            less=0;%y'的目标函数值小于个体的目标函数值数目
            equal=0;%y'的目标函数值等于个体的目标函数值数目
            greater=0;%y'的目标函数值大于个体的目标函数值数目
            for mm=1:f_num
                if(off(:,x_num+mm)>EP(k,x_num+mm))
                    greater=greater+1;
                elseif(off(:,x_num+mm)==EP(k,x_num+mm))
                    equal=equal+1;
                else
                    less=less+1;
                end
            end
            %%%从EP中移除被y'支配的向量
            if(greater==0 && equal~=f_num)
                kk=[k kk];
            end
            %%%如果EP中没有支配y'的个体，将y'加入EP
            if(less==0 && equal~=f_num)
                temp=1;
            end
        end
        if(isempty(kk)==0)
            EP(kk,:)=[];
        end
        if(temp==0)
            EP=[EP;off];
        end
    end
    if mod(gen,10) == 0
        fprintf('%d gen has completed!\n',gen);
    end
%     if f_num==2
%         plot(EP(:,x_num+1),EP(:,x_num+2),'r*');
%     elseif f_num==3
%         plot3( EP(:,x_num+1), EP(:,x_num+2),EP(:,x_num+3),'r*' );
%         set(gca,'xdir','reverse'); set(gca,'ydir','reverse');
%     end
%     title(num2str(gen));
%     drawnow
end
filepath=pwd;          
cd('E:\GA\MOEA D\MOEAD_王超\MOEAD王超(EP版)\DTLZ2');
save solution5.txt EP -ASCII
cd(filepath);
toc;
%------------------------画图对比--------------------------
if f_num==2
    hold on
    plot(EP(:,x_num+1),EP(:,x_num+2),'r*');
elseif f_num==3
    hold on
    plot3( EP(:,x_num+1), EP(:,x_num+2),EP(:,x_num+3),'r*' );
    set(gca,'xdir','reverse'); set(gca,'ydir','reverse');
end
% figure;
% if f_num==2
%     hold on
%     plot(X(:,x_num+1),X(:,x_num+2),'r*');
% elseif f_num==3
%     hold on
%     plot3( X(:,x_num+1), X(:,x_num+2),X(:,x_num+3),'r*' );
%     set(gca,'xdir','reverse'); set(gca,'ydir','reverse');
% end
%--------------------Coverage(C-metric)---------------------
A=PP;B=EP(:,(x_num+1):(x_num+f_num));%%%%%%%%%%%%%%%%%%%%
[temp_A,~]=size(A);
[temp_B,~]=size(B);
number=0;
for i=1:temp_B
    nn=0;
    for j=1:temp_A
        less=0;%当前个体的目标函数值小于多少个体的数目
        equal=0;%当前个体的目标函数值等于多少个体的数目
        for k=1:f_num
            if(B(i,k)<A(j,k))
                less=less+1;
            elseif(B(i,k)==A(j,k))
                equal=equal+1;
            end
        end
        if(less==0 && equal~=f_num)
            nn=nn+1;%被支配个体数目n+1
        end
    end
    if(nn~=0)
        number=number+1;
    end
end
C_AB=number/temp_B;
disp("C_AB:");
disp(C_AB);
%-----Distance from Representatives in the PF(D-metric)-----
A=EP(:,(x_num+1):(x_num+f_num));P=PP;%%%%%%%%%%%%%%%%%%%
[temp_A,~]=size(A);
[temp_P,~]=size(P);
min_d=0;
for v=1:temp_P
    d_va=(A-repmat(P(v,:),temp_A,1)).^2;
    min_d=min_d+min(sqrt(sum(d_va,2)));
end
D_AP=(min_d/temp_P);
disp("D_AP:");
disp(D_AP);
filepath=pwd;          
cd('E:\GA\MOEA D\MOEAD_王超\MOEAD王超(EP版)\DTLZ2');
save C_AB5.txt C_AB -ASCII
save D_AP5.txt D_AP -ASCII
cd(filepath);