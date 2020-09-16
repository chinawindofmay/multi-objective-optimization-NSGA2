function [F,chromo] = non_domination_sort( pop,chromo,f_num,x_num )
%non_domination_sort 初始种群的非支配排序和计算拥挤度
%初始化pareto等级为1
pareto_rank=1;
F(pareto_rank).ss=[];%pareto等级为pareto_rank的集合
p=[];%每个个体p的集合
for i=1:pop
    %%%计算出种群中每个个体p的被支配个数n和该个体支配的解的集合s
    p(i).n=0;%被支配个体数目n
    p(i).s=[];%支配解的集合s
    for j=1:pop
        less=0;%y'的目标函数值小于个体的目标函数值数目
        equal=0;%y'的目标函数值等于个体的目标函数值数目
        greater=0;%y'的目标函数值大于个体的目标函数值数目
        for k=1:f_num
            if(chromo(i,x_num+k)<chromo(j,x_num+k))
                less=less+1;
            elseif(chromo(i,x_num+k)==chromo(j,x_num+k))
                equal=equal+1;
            else
                greater=greater+1;
            end
        end
        if(less==0 && equal~=f_num)%if(less==0 && greater~=0)
            p(i).n=p(i).n+1;%被支配个体数目n+1
        elseif(greater==0 && equal~=f_num)%elseif(greater==0 && less~=0)
            p(i).s=[p(i).s j];
        end
    end
    %%%将种群中参数为n的个体放入集合F(1)中
    if(p(i).n==0)
        chromo(i,f_num+x_num+1)=1;%储存个体的等级信息
        F(pareto_rank).ss=[F(pareto_rank).ss i];
    end
end
%%%求pareto等级为pareto_rank+1的个体
while ~isempty(F(pareto_rank).ss)
    temp=[];
    for i=1:length(F(pareto_rank).ss)
        if ~isempty(p(F(pareto_rank).ss(i)).s)
            for j=1:length(p(F(pareto_rank).ss(i)).s)
                p(p(F(pareto_rank).ss(i)).s(j)).n=p(p(F(pareto_rank).ss(i)).s(j)).n - 1;%nl=nl-1
                if p(p(F(pareto_rank).ss(i)).s(j)).n==0
                    chromo(p(F(pareto_rank).ss(i)).s(j),f_num+x_num+1)=pareto_rank+1;%储存个体的等级信息
                    temp=[temp p(F(pareto_rank).ss(i)).s(j)];
                end
            end
        end
    end
    pareto_rank=pareto_rank+1;
    F(pareto_rank).ss=temp;
end