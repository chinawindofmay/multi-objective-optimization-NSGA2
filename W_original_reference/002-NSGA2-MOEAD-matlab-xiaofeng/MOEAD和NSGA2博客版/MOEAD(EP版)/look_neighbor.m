function B = look_neighbor( lamda,T )
%������������Ȩ���������ŷʽ����
N =size(lamda,1);
B=zeros(N,T);
distance=zeros(N,N);
for i=1:N
    for j=1:N
        l=lamda(i,:)-lamda(j,:);
        distance(i,j)=sqrt(l*l');
    end
end
%����ÿ��Ȩ���������T��Ȩ������������
for i=1:N
    [~,index]=sort(distance(i,:));
    B(i,:)=index(1:T);
end

