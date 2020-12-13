function chromo_offspring = cross_mutation2( chromo_parent_1,chromo_parent_2,f_num,x_num,x_min,x_max,pc,pm,yita1,yita2,fun )
%模拟二进制交叉与多项式变异
%%%模拟二进制交叉
if(rand(1)<pc)
    %初始化子代种群
    off_1=zeros(1,x_num+f_num);
    %进行模拟二进制交叉
    gama=zeros(1,x_num);
    for j=1:x_num
        u1=rand;
        if u1<=0.5
            gama(j)=(2*u1)^(1/(yita1+1));
        else
            gama(j)=(1/(2*(1-u1)))^(1/(yita1+1));
        end
        off_1(j)=0.5*((1-gama(j))*chromo_parent_1(j)+(1+gama(j))*chromo_parent_2(j));
        %使子代在定义域内
        if(off_1(j)>x_max(j))
            off_1(j)=x_max(j);
        elseif(off_1(j)<x_min(j))
            off_1(j)=x_min(j);
        end
    end
    %计算子代个体的目标函数值
    off_1(1,(x_num+1):(x_num+f_num))=object_fun(off_1,f_num,x_num,fun);
end
%%%多项式变异
if(rand(1)<pm)
    r=randperm(x_num);
    ind=r(1);        %选中变异的位置
    r=rand; 
    if r<0.5
        delta=(2*r)^(1/(1+yita2))-1;
    else
        delta=1-(2*(1-r))^(1/(yita2+1));
    end
    off_1(ind)=off_1(ind)+delta*(x_max(ind)-x_min(ind));
    for j=1:x_num
        %使子代在定义域内
        if(off_1(j)>x_max(j))
            off_1(j)=x_max(j);
        elseif(off_1(j)<x_min(j))
            off_1(j)=x_min(j);
        end
    end
    %计算子代个体的目标函数值
    off_1(1,(x_num+1):(x_num+f_num))=object_fun(off_1,f_num,x_num,fun);
end
chromo_offspring=off_1;
end

