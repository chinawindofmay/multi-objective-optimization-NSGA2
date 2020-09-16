%---------------------------------------------------------------------
%�����ܣ�ʵ��MOEAD�㷨�����Ժ���ΪZDT1,ZDT2,ZDT3,ZDT4,ZDT6,DTLZ1,DTLZ2
%˵�����Ŵ�����Ϊģ������ƽ���Ͷ���ʽ����
%���ߣ�(����)
%email: 18821709267@163.com 
%�������ʱ�䣺2018.09.30
%����޸�ʱ�䣺2018.10.08
%�ο����ģ�
%MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition
%Qingfu Zhang, Senior Member, IEEE, and Hui Li
%IEEE TRANSACTIONS O
%----------------------------------------------------------
clear all
clc
tic;
%------------------------��������--------------------------
format long
global x_max x_min x_num f_num lamda z
rand('state',sum(100*clock));
N=300;%��Ⱥ��С
T=20;%�ھӹ�ģ��С
fun='DTLZ1';%���Ժ���
funfun;%���Ժ���
lamda=genrate_lamda(N,f_num);%���ȷֲ���N��Ȩ������
max_gen=250;%��������
pc=1;%�������
pm=1/x_num;%�������
yita1=2;%ģ������ƽ������2
yita2=5;%����ʽ�������5
%------------------------��ʼ����--------------------------
%%������������Ȩ���������ŷʽ���룬����ÿ��Ȩ���������T��Ȩ������������
B=look_neighbor(lamda,T);
%%�ڿ��пռ�������������ʼ��Ⱥ
X=initialize(N,f_num,x_num,x_min,x_max,fun);
%%��ʼ��z
for i=1:f_num
    z(i) = min(X(:,x_num+i));
end
%------------------------��������--------------------------
for gen=1:max_gen
    for i=1:N
        %%�������飬��B(i)�����ѡȡ��������k��l
        index1 = randperm(T);
        parent1 = B(i,index1(1));
        parent2 = B(i,index1(2));
        off=cross_mutation(X(parent1,:),X(parent2,:),f_num,x_num,x_min,x_max,pc,pm,yita1,yita2,fun );
        %off=cross_mutation2(X(parent1,:),X(parent2,:),f_num,x_num,x_min,x_max,pc,pm,yita1,yita2,fun );
        %%����z
        for j=1:f_num
            %%if(Zi<fi(y')),zi=fi(y')
            z(j)=min(z(j),off(:,x_num+j));
        end
        %%���������
        X=updateNeighbor(lamda,z,X,B(i,:),off,x_num,f_num);
    end
    if mod(gen,10) == 0
        fprintf('%d gen has completed!\n',gen);
    end
%     if f_num==2
%         plot(X(:,x_num+1),X(:,x_num+2),'r*');
%     elseif f_num==3
%         plot3( X(:,x_num+1), X(:,x_num+2),X(:,x_num+3),'r*' );
%         set(gca,'xdir','reverse'); set(gca,'ydir','reverse');
%     end
%     title(num2str(gen));
%     drawnow
end
toc;
aaa=toc;
%------------------------��ͼ�Ա�--------------------------
if f_num==2
    hold on
    plot(X(:,x_num+1),X(:,x_num+2),'r*');
elseif f_num==3
    hold on
    plot3( X(:,x_num+1), X(:,x_num+2),X(:,x_num+3),'r*' );
    set(gca,'xdir','reverse'); set(gca,'ydir','reverse');
end
%--------------------Coverage(C-metric)---------------------
A=PP;B=X(:,(x_num+1):(x_num+f_num));%%%%%%%%%%%%%%%%%%%
[temp_A,~]=size(A);
[temp_B,~]=size(B);
number=0;
for i=1:temp_B
    nn=0;
    for j=1:temp_A
        less=0;%��ǰ�����Ŀ�꺯��ֵС�ڶ��ٸ������Ŀ
        equal=0;%��ǰ�����Ŀ�꺯��ֵ���ڶ��ٸ������Ŀ
        for k=1:f_num
            if(B(i,k)<A(j,k))
                less=less+1;
            elseif(B(i,k)==A(j,k))
                equal=equal+1;
            end
        end
        if(less==0 && equal~=f_num)
            nn=nn+1;%��֧�������Ŀn+1
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
A=X(:,(x_num+1):(x_num+f_num));P=PP;%%%%%%�����Ĺ�ʽ������ͬ��ע��A��������һ��
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
cd('E:\GA\MOEA D\MOEAD_����\MOEAD����\DTLZ1');
save solution4.txt X -ASCII
save C_AB4.txt C_AB -ASCII
save D_AP4.txt D_AP -ASCII
save toc4.txt aaa -ASCII
cd(filepath);