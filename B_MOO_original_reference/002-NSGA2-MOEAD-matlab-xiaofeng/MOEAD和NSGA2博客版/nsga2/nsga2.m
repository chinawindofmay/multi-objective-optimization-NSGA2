%---------------------------------------------------------------------
%�����ܣ�ʵ��nsga2�㷨�����Ժ���ΪZDT1,ZDT2,ZDT3,ZDT4,ZDT6,DTLZ1,DTLZ2
%˵�����Ŵ�����Ϊ�����ƾ���ѡ��ģ������ƽ���Ͷ���ʽ����
%      ��Ҫ���õĲ�����pop,gen,pc,pm,yita1,yita2
%���ߣ�(����)
%�������ʱ�䣺2018.9.3
%����޸�ʱ�䣺2018.9.20
%----------------------------------------------------------
clear all
clc
tic;
%��������
fun='DTLZ1';
funfun;%����ѡ��
pop=300;%��Ⱥ��С100
gen=250;%250��������
pc=1;%�������
pm=1/x_num;%�������
% yita1=1;%ģ������ƽ������
% yita2=10;%����ʽ�������
yita1=2;%ģ������ƽ������10
yita2=5;%����ʽ�������50

%%��ʼ����Ⱥ
chromo=initialize(pop,f_num,x_num,x_min,x_max,fun);
%%��ʼ��Ⱥ�ķ�֧������
[F1,chromo_non]=non_domination_sort(pop,chromo,f_num,x_num);%FΪpareto�ȼ�Ϊpareto_rank�ļ��ϣ�%pΪÿ������p�ļ���(����ÿ������p�ı�֧�����n�͸ø���֧��Ľ�ļ���s),chromo���һ�м������ĵȼ�
%%����ӵ���Ƚ�������
chromo=crowding_distance_sort(F1,chromo_non,f_num,x_num);
i=1;
%%ѭ����ʼ
for i=1:gen
    %%�����ƾ���ѡ��(kȡ��pop/2������ѡ����)
    chromo_parent_1 = tournament_selection(chromo);
    chromo_parent_2 = tournament_selection(chromo);
    chromo_parent=[chromo_parent_1;chromo_parent_2];
    %%ģ������ƽ��������ʽ����
    chromo_offspring=cross_mutation(chromo_parent,f_num,x_num,x_min,x_max,pc,pm,yita1,yita2,fun );
    %%��Ӣ��������
    %���������Ӵ��ϲ�
    [pop_parent,~]=size(chromo);
    [pop_offspring,~]=size(chromo_offspring);
    combine_chromo(1:pop_parent,1:(f_num+x_num))=chromo(:,1:(f_num+x_num));
    combine_chromo((pop_parent+1):(pop_parent+pop_offspring),1:(f_num+x_num))=chromo_offspring(:,1:(f_num+x_num));
    %���ٷ�֧������
    [pop2,~]=size(combine_chromo);
    [F2,combine_chromo1]=non_domination_sort(pop2,combine_chromo,f_num,x_num);
    %����ӵ���Ƚ�������
    combine_chromo2=crowding_distance_sort(F2,combine_chromo1,f_num,x_num);
    %��Ӣ����������һ����Ⱥ
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
%%--------------------delta����--------------------------------
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
A=chromo(:,(x_num+1):(x_num+f_num));P=PP;%%%%%%�����Ĺ�ʽ������ͬ��ע��A��������һ��
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
cd('E:\GA\MOEA D\MOEAD_����\nsga2�Ա������޸�\DTLZ1');
save C_AB4.txt C_AB -ASCII
save D_AP4.txt D_AP -ASCII
save solution4.txt chromo -ASCII
save toc4.txt aaa -ASCII
cd(filepath);