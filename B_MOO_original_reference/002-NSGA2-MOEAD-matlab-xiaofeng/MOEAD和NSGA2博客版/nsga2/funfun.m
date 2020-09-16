%测试函数
%--------------------ZDT1--------------------
if strcmp(fun,'ZDT1')
    f_num=2;%目标函数个数
    x_num=30;%决策变量个数
    x_min=zeros(1,x_num);%决策变量的最小值
    x_max=ones(1,x_num);%决策变量的最大值
    load('ZDT1.txt');%导入正确的前端解
    plot(ZDT1(:,1),ZDT1(:,2),'g*');
    PP=ZDT1;
end
%--------------------ZDT2--------------------
if strcmp(fun,'ZDT2')
    f_num=2;%目标函数个数
    x_num=30;%决策变量个数
    x_min=zeros(1,x_num);%决策变量的最小值
    x_max=ones(1,x_num);%决策变量的最大值
    load('ZDT2.txt');%导入正确的前端解
    plot(ZDT2(:,1),ZDT2(:,2),'g*');
    PP=ZDT2;
end
%--------------------ZDT3--------------------
if strcmp(fun,'ZDT3')
    f_num=2;%目标函数个数
    x_num=30;%决策变量个数
    x_min=zeros(1,x_num);%决策变量的最小值
    x_max=ones(1,x_num);%决策变量的最大值
    load('ZDT3.txt');%导入正确的前端解
    plot(ZDT3(:,1),ZDT3(:,2),'g*');
    PP=ZDT3;
end
%--------------------ZDT4--------------------
if strcmp(fun,'ZDT4')
    f_num=2;%目标函数个数
    x_num=10;%决策变量个数
    x_min=[0,-5,-5,-5,-5,-5,-5,-5,-5,-5];%决策变量的最小值
    x_max=[1,5,5,5,5,5,5,5,5,5];%决策变量的最大值
    load('ZDT4.txt');%导入正确的前端解
    plot(ZDT4(:,1),ZDT4(:,2),'g*');
    PP=ZDT4;
end
%--------------------ZDT6--------------------
if strcmp(fun,'ZDT6')
    f_num=2;%目标函数个数
    x_num=10;%决策变量个数
    x_min=zeros(1,x_num);%决策变量的最小值
    x_max=ones(1,x_num);%决策变量的最大值
    load('ZDT6.txt');%导入正确的前端解
    plot(ZDT6(:,1),ZDT6(:,2),'g*');
    PP=ZDT6;
end
%--------------------------------------------
%--------------------DTLZ1--------------------
if strcmp(fun,'DTLZ1')
    f_num=3;%目标函数个数
    x_num=10;%决策变量个数
    x_min=zeros(1,x_num);%决策变量的最小值
    x_max=ones(1,x_num);%决策变量的最大值
    load('DTLZ1.txt');%导入正确的前端解
%     plot3(DTLZ1(:,1),DTLZ1(:,2),DTLZ1(:,3),'g*');
    PP=DTLZ1;
end
%--------------------------------------------
%--------------------DTLZ2--------------------
if strcmp(fun,'DTLZ2')
    f_num=3;%目标函数个数
    x_num=10;%决策变量个数
    x_min=zeros(1,x_num);%决策变量的最小值
    x_max=ones(1,x_num);%决策变量的最大值
    load('DTLZ2.txt');%导入正确的前端解
%     plot3(DTLZ2(:,1),DTLZ2(:,2),DTLZ2(:,3),'g*');
    PP=DTLZ2;
end
%--------------------------------------------