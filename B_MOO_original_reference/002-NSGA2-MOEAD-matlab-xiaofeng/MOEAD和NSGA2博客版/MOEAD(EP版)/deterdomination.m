function X = deterdomination( X,N,f_num,x_num )
%��ʼ���Ƿ�Ϊ��֧�����
for i=1:N
    X(i,(x_num+f_num+1))=0;
end

for i=1:N
    nn=0;
    for j=1:N
        less=0;%��ǰ�����Ŀ�꺯��ֵС�ڶ��ٸ������Ŀ
        equal=0;%��ǰ�����Ŀ�꺯��ֵ���ڶ��ٸ������Ŀ
        for k=1:f_num
            if(X(i,x_num+k)<X(j,x_num+k))
                less=less+1;
            elseif(X(i,x_num+k)==X(j,x_num+k))
                equal=equal+1;
            end
        end
        if(less==0 && equal~=f_num)
            nn=nn+1;%��֧�������Ŀn+1
        end
    end
    if(nn==0)
        X(i,(x_num+f_num+1))=1;
    end
end
end

