function chromo = initialize( pop,f_num,x_num,x_min,x_max,fun )
%   种群初始化
chromo = repmat(x_min,pop,1)+rand(pop,x_num).*repmat(x_max-x_min,pop,1); 
for i=1:pop
    chromo(i,(x_num+1:(x_num+f_num))) = object_fun(chromo(i,:),f_num,x_num,fun);
    chromo(i,(x_num+f_num+1)) = 0;
end
% for i=1:pop
%     for j=1:x_num
%         chromo(i,j)=x_min(j)+(x_max(j)-x_min(j))*rand(1);
%     end
%     chromo(i,(x_num+1:(x_num+f_num))) = object_fun(chromo(i,:),f_num,x_num,fun);
% end