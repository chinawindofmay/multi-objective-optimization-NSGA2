function chromo = crowding_distance_sort( F,chromo,f_num,x_num )
%计算拥挤度
%%%按照pareto等级对种群中的个体进行排序
[~,index]=sort(chromo(:,f_num+x_num+1));
[~,mm1]=size(chromo);
temp=zeros(length(index),mm1);
for i=1:length(index)%=pop
    temp(i,:)=chromo(index(i),:);%按照pareto等级排序后种群
end

%%%对于每个等级的个体开始计算拥挤度
current_index = 0;
for pareto_rank=1:(length(F)-1)%计算F的循环时多了一次空，所以减掉
    %%拥挤度初始化为0
    nd=[];
    nd(:,1)=zeros(length(F(pareto_rank).ss),1);
    %y=[];%储存当前处理的等级的个体
    [~,mm2]=size(temp);
    y=zeros(length(F(pareto_rank).ss),mm2);%储存当前处理的等级的个体
    previous_index=current_index + 1;
    for i=1:length(F(pareto_rank).ss)
        y(i,:)=temp(current_index + i,:);
    end
    current_index=current_index + i;
    %%对于每一个目标函数fm
    for i=1:f_num
        %%根据该目标函数值对该等级的个体进行排序
        [~,index_objective]=sort(y(:,x_num+i));
        %objective_sort=[];%通过目标函数排序后的个体
        [~,mm3]=size(y);
        objective_sort=zeros(length(index_objective),mm3);%通过目标函数排序后的个体
        for j=1:length(index_objective)
            objective_sort(j,:)=y(index_objective(j),:);
        end
        %%记fmax为最大值，fmin为最小值
        fmin=objective_sort(1,x_num+i);
        fmax=objective_sort(length(index_objective),x_num+i);
        %%对排序后的两个边界拥挤度设为1d和nd设为无穷
        y(index_objective(1),f_num+x_num+1+i)=Inf;
        y(index_objective(length(index_objective)),f_num+x_num+1+i)=Inf;
        %%计算nd=nd+(fm(i+1)-fm(i-1))/(fmax-fmin)
        for j=2:(length(index_objective)-1)
            pre_f=objective_sort(j-1,x_num+i);
            next_f=objective_sort(j+1,x_num+i);
            if (fmax-fmin==0)
                y(index_objective(j),f_num+x_num+1+i)=Inf;
            else
                y(index_objective(j),f_num+x_num+1+i)=(next_f-pre_f)/(fmax-fmin);
            end
        end
    end
    %多个目标函数拥挤度求和
    for i=1:f_num
        nd(:,1)=nd(:,1)+y(:,f_num+x_num+1+i);
    end
    %第2列保存拥挤度，其他的覆盖掉
    y(:,f_num+x_num+2)=nd;
    y=y(:,1:(f_num+x_num+2));
    temp_two(previous_index:current_index,:) = y;
end
chromo=temp_two;
end
