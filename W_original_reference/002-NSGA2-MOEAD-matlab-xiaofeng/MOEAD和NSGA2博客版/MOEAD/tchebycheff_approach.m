function fs = tchebycheff_approach( lamda,z,f,i)
%tchebycheff_approach
for j=1:length(lamda(i,:))
    if(lamda(i,j)==0)
        lamda(i,j)=0.00001;
    end
end
fs=max(lamda(i,:).*abs(f-z));
end

