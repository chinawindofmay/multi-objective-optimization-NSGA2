function chromo = elitism( pop,combine_chromo2,f_num,x_num )
%精英保留策略
[pop2,temp]=size(combine_chromo2);
chromo_rank=zeros(pop2,temp);
chromo=zeros(pop,(f_num+x_num+2));
%根据pareto等级从高到低进行排序
[~,index_rank]=sort(combine_chromo2(:,f_num+x_num+1));
for i=1:pop2
    chromo_rank(i,:)=combine_chromo2(index_rank(i),:);
end
%求出最高的pareto等级
max_rank=max(combine_chromo2(:,f_num+x_num+1));
%根据排序后的顺序，将等级相同的种群整个放入父代种群中，直到某一层不能全部放下为止
prev_index=0;
for i=1:max_rank
    %寻找当前等级i个体里的最大索引
    current_index=max(find(chromo_rank(:,(f_num+x_num+1))==i));
    %不能放下为止
    if(current_index>pop)
        %剩余群体数
        remain_pop=pop-prev_index;
        %等级为i的个体
        temp=chromo_rank(((prev_index+1):current_index),:);
        %根据拥挤度从大到小排序
        [~,index_crowd]=sort(temp(:,(f_num+x_num+2)),'descend');
        %填满父代
        for j=1:remain_pop
            chromo(prev_index+j,:)=temp(index_crowd(j),:);
        end
        return;
    elseif(current_index<pop)
        % 将所有等级为i的个体直接放入父代种群
        chromo(((prev_index+1):current_index),:)=chromo_rank(((prev_index+1):current_index),:);
    else
        % 将所有等级为i的个体直接放入父代种群
        chromo(((prev_index+1):current_index),:)=chromo_rank(((prev_index+1):current_index),:);
        return;
    end
    prev_index = current_index;
end
end

