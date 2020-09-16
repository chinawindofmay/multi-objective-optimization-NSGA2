function X = updateNeighbor( lamda,z,X,Bi,off,x_num,f_num )
%更新领域解
for i=1:length(Bi)
    gte_xi=tchebycheff_approach(lamda,z,X(Bi(i),(x_num+1):(x_num+f_num)),Bi(i));
    gte_off=tchebycheff_approach(lamda,z,off(:,(x_num+1):(x_num+f_num)),Bi(i));
%     gte_xi=ws_approach(lamda,X(Bi(i),(x_num+1):(x_num+f_num)),Bi(i));
%     gte_off=ws_approach(lamda,off(:,(x_num+1):(x_num+f_num)),Bi(i));
    if gte_off <= gte_xi
        X(Bi(i),:)=off;
    end
end

