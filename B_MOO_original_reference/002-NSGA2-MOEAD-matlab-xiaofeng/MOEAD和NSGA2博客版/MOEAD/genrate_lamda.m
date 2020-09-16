function lamda = genrate_lamda( N,f_num )
%产生初始化向量lamda
lamda2=zeros(N+1,f_num);%初始化
if f_num==2
    array=(0:N)/N;%均匀分布的值
    for i=1:N+1
            lamda2(i,1)=array(i);
            lamda2(i,2)=1-array(i);
    end
    len = size(lamda2,1);
    index = randperm(len);
    index = sort(index(1:N));
    lamda = lamda2(index,:);
elseif f_num==3
    k = 1;
    array = (0:25)/25;%产生均匀分布的值
    for i=1:26
        for j = 1:26
            if i+j<28
                lamda3(k,1) = array(i);
                lamda3(k,2) = array(j);
                lamda3(k,3) = array(28-i-j);
                k=k+1;
            end
        end
    end
    len = size(lamda3,1);
    index = randperm(len);
    index = sort(index(1:N));
    lamda = lamda3(index,:);
end
end

