function B = look_neighbor( lamda,T )
%计算任意两个权重向量间的欧式距离
N =size(lamda,1);
B=zeros(N,T);
distance=zeros(N,N);
for i=1:N
    for j=1:N
        l=lamda(i,:)-lamda(j,:);
        distance(i,j)=sqrt(l*l');
    end
end
%查找每个权向量最近的T个权重向量的索引
for i=1:N
    [~,index]=sort(distance(i,:));
    B(i,:)=index(1:T);
end

