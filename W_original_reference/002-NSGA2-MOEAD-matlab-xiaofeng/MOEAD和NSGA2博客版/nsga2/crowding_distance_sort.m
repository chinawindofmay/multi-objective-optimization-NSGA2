function chromo = crowding_distance_sort( F,chromo,f_num,x_num )
%����ӵ����
%%%����pareto�ȼ�����Ⱥ�еĸ����������
[~,index]=sort(chromo(:,f_num+x_num+1));
[~,mm1]=size(chromo);
temp=zeros(length(index),mm1);
for i=1:length(index)%=pop
    temp(i,:)=chromo(index(i),:);%����pareto�ȼ��������Ⱥ
end

%%%����ÿ���ȼ��ĸ��忪ʼ����ӵ����
current_index = 0;
for pareto_rank=1:(length(F)-1)%����F��ѭ��ʱ����һ�οգ����Լ���
    %%ӵ���ȳ�ʼ��Ϊ0
    nd=[];
    nd(:,1)=zeros(length(F(pareto_rank).ss),1);
    %y=[];%���浱ǰ����ĵȼ��ĸ���
    [~,mm2]=size(temp);
    y=zeros(length(F(pareto_rank).ss),mm2);%���浱ǰ����ĵȼ��ĸ���
    previous_index=current_index + 1;
    for i=1:length(F(pareto_rank).ss)
        y(i,:)=temp(current_index + i,:);
    end
    current_index=current_index + i;
    %%����ÿһ��Ŀ�꺯��fm
    for i=1:f_num
        %%���ݸ�Ŀ�꺯��ֵ�Ըõȼ��ĸ����������
        [~,index_objective]=sort(y(:,x_num+i));
        %objective_sort=[];%ͨ��Ŀ�꺯�������ĸ���
        [~,mm3]=size(y);
        objective_sort=zeros(length(index_objective),mm3);%ͨ��Ŀ�꺯�������ĸ���
        for j=1:length(index_objective)
            objective_sort(j,:)=y(index_objective(j),:);
        end
        %%��fmaxΪ���ֵ��fminΪ��Сֵ
        fmin=objective_sort(1,x_num+i);
        fmax=objective_sort(length(index_objective),x_num+i);
        %%�������������߽�ӵ������Ϊ1d��nd��Ϊ����
        y(index_objective(1),f_num+x_num+1+i)=Inf;
        y(index_objective(length(index_objective)),f_num+x_num+1+i)=Inf;
        %%����nd=nd+(fm(i+1)-fm(i-1))/(fmax-fmin)
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
    %���Ŀ�꺯��ӵ�������
    for i=1:f_num
        nd(:,1)=nd(:,1)+y(:,f_num+x_num+1+i);
    end
    %��2�б���ӵ���ȣ������ĸ��ǵ�
    y(:,f_num+x_num+2)=nd;
    y=y(:,1:(f_num+x_num+2));
    temp_two(previous_index:current_index,:) = y;
end
chromo=temp_two;
end
