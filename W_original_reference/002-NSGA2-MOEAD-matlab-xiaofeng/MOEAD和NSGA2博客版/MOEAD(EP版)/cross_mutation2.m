function chromo_offspring = cross_mutation2( chromo_parent_1,chromo_parent_2,f_num,x_num,x_min,x_max,pc,pm,yita1,yita2,fun )
%ģ������ƽ��������ʽ����
%%%ģ������ƽ���
if(rand(1)<pc)
    %��ʼ���Ӵ���Ⱥ
    off_1=zeros(1,x_num+f_num);
    %����ģ������ƽ���
    gama=zeros(1,x_num);
    for j=1:x_num
        u1=rand;
        if u1<=0.5
            gama(j)=(2*u1)^(1/(yita1+1));
        else
            gama(j)=(1/(2*(1-u1)))^(1/(yita1+1));
        end
        off_1(j)=0.5*((1-gama(j))*chromo_parent_1(j)+(1+gama(j))*chromo_parent_2(j));
        %ʹ�Ӵ��ڶ�������
        if(off_1(j)>x_max(j))
            off_1(j)=x_max(j);
        elseif(off_1(j)<x_min(j))
            off_1(j)=x_min(j);
        end
    end
    %�����Ӵ������Ŀ�꺯��ֵ
    off_1(1,(x_num+1):(x_num+f_num))=object_fun(off_1,f_num,x_num,fun);
end
%%%����ʽ����
if(rand(1)<pm)
    r=randperm(x_num);
    ind=r(1);        %ѡ�б����λ��
    r=rand; 
    if r<0.5
        delta=(2*r)^(1/(1+yita2))-1;
    else
        delta=1-(2*(1-r))^(1/(yita2+1));
    end
    off_1(ind)=off_1(ind)+delta*(x_max(ind)-x_min(ind));
    for j=1:x_num
        %ʹ�Ӵ��ڶ�������
        if(off_1(j)>x_max(j))
            off_1(j)=x_max(j);
        elseif(off_1(j)<x_min(j))
            off_1(j)=x_min(j);
        end
    end
    %�����Ӵ������Ŀ�꺯��ֵ
    off_1(1,(x_num+1):(x_num+f_num))=object_fun(off_1,f_num,x_num,fun);
end
chromo_offspring=off_1;
end

