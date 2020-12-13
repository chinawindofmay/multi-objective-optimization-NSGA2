%---------------------------------------------------------------------
%程序功能：实现nsga2算法，测试函数为ZDT1,ZDT2,ZDT3,ZDT4,ZDT6,DTLZ1,DTLZ2
%说明：遗传算子为二进制竞赛选择，模拟二进制交叉和多项式变异
%      需要设置的参数：pop,gen,pc,pm,yita1,yita2
%作者：(晓风)
%最初建立时间：2018.9.3
%最近修改时间：2018.9.20
%----------------------------------------------------------
clear all
clc
tic;
%参数设置
fun='DTLZ1';
funfun;%函数选择
pop=300;%种群大小100
gen=250;%250进化代数
pc=1;%交叉概率
pm=1/x_num;%变异概率
% yita1=1;%模拟二进制交叉参数
% yita2=10;%多项式变异参数
yita1=2;%模拟二进制交叉参数10
yita2=5;%多项式变异参数50

%%初始化种群
chromo=initialize(pop,f_num,x_num,x_min,x_max,fun);
%%初始种群的非支配排序
[F1,chromo_non]=non_domination_sort(pop,chromo,f_num,x_num);%F为pareto等级为pareto_rank的集合，%p为每个个体p的集合(包括每个个体p的被支配个数n和该个体支配的解的集合s),chromo最后一列加入个体的等级
%%计算拥挤度进行排序
chromo=crowding_distance_sort(F1,chromo_non,f_num,x_num);
i=1;
%%循环开始
for i=1:gen
    %%二进制竞赛选择(k取了pop/2，所以选两次)
    chromo_parent_1 = tournament_selection(chromo);
    chromo_parent_2 = tournament_selection(chromo);
    chromo_parent=[chromo_parent_1;chromo_parent_2];
    %%模拟二进制交叉与多项式变异
    chromo_offspring=cross_mutation(chromo_parent,f_num,x_num,x_min,x_max,pc,pm,yita1,yita2,fun );
    %%精英保留策略
    %将父代和子代合并
    [pop_parent,~]=size(chromo);
    [pop_offspring,~]=size(chromo_offspring);
    combine_chromo(1:pop_parent,1:(f_num+x_num))=chromo(:,1:(f_num+x_num));
    combine_chromo((pop_parent+1):(pop_parent+pop_offspring),1:(f_num+x_num))=chromo_offspring(:,1:(f_num+x_num));
    %快速非支配排序
    [pop2,~]=size(combine_chromo);
    [F2,combine_chromo1]=non_domination_sort(pop2,combine_chromo,f_num,x_num);
    %计算拥挤度进行排序
    combine_chromo2=crowding_distance_sort(F2,combine_chromo1,f_num,x_num);
    %精英保留产生下一代种群
    chromo=elitism(pop,combine_chromo2,f_num,x_num);
    if mod(i,10) == 0
        fprintf('%d gen has completed!\n',i);
    end
end
toc;
aaa=toc;

hold on
if(f_num==2)
    plot(chromo(:,x_num+1),chromo(:,x_num+2),'r*');
end
if(f_num==3)
    plot3(chromo(:,x_num+1),chromo(:,x_num+2),chromo(:,x_num+2),'r*');
end
%%--------------------delta度量--------------------------------
% [~,index_f1]=sort(chromo(:,x_num+1));
% d=zeros(length(chromo)-1,1);
% for i=1:(length(chromo)-1)
%     d(i)=((chromo(index_f1(i),x_num+1)-chromo(index_f1(i+1),x_num+1))^2+(chromo(index_f1(i),x_num+2)-chromo(index_f1(i+1),x_num+2))^2)^0.5;
% end
% d_mean1=mean(d);
% d_mean=d_mean1*ones(length(chromo)-1,1);
% d_sum=sum(abs(d-d_mean));
% delta=(d(1)+d(length(chromo)-1)+d_sum)/(d(1)+d(length(chromo)-1)+(length(chromo)-1)*d_mean1);
% disp(delta);
%mean(a)
%(std(a))^2
%--------------------Coverage(C-metric)---------------------
A=PP;B=chromo(:,(x_num+1):(x_num+f_num));%%%%%%%%%%%%%%%%%%%
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
A=chromo(:,(x_num+1):(x_num+f_num));P=PP;%%%%%%与论文公式保持相同，注意A和上述不一样
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
cd('E:\GA\MOEA D\MOEAD_王超\nsga2对比算子修改\DTLZ1');
save C_AB4.txt C_AB -ASCII
save D_AP4.txt D_AP -ASCII
save solution4.txt chromo -ASCII
save toc4.txt aaa -ASCII
cd(filepath);