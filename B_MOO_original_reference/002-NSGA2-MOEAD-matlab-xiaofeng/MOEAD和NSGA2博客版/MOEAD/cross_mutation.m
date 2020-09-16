function chromo_offspring = cross_mutation( chromo_parent_1,chromo_parent_2,f_num,x_num,x_min,x_max,pc,pm,yita1,yita2,fun )
%ģ������ƽ��������ʽ����
%%%ģ������ƽ���
if(rand(1)<pc)
    %��ʼ���Ӵ���Ⱥ
    off_1=zeros(1,x_num+f_num);
    %����ģ������ƽ���
    u1=zeros(1,x_num);
    gama=zeros(1,x_num);
    for j=1:x_num
        u1(j)=rand(1);
        if u1(j)<=0.5
            gama(j)=(2*u1(j))^(1/(yita1+1));
        else
            gama(j)=(1/(2*(1-u1(j))))^(1/(yita1+1));
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
    u2=zeros(1,x_num);
    delta=zeros(1,x_num);
    for j=1:x_num
        u2(j)=rand(1);
        if(u2(j)<0.5)
            delta(j)=(2*u2(j))^(1/(yita2+1))-1;
        else
            delta(j)=1-(2*(1-u2(j)))^(1/(yita2+1));
        end
        off_1(j)=off_1(j)+delta(j);
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

