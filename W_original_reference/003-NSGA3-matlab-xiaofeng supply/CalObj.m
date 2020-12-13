function PopObj = CalObj(PopDec)
global M D name
    NN = size(PopDec,1);
    switch name
        case 'DTLZ1'
            g      = 100*(D-M+1+sum((PopDec(:,M:end)-0.5).^2-cos(20.*pi.*(PopDec(:,M:end)-0.5)),2));
            PopObj = 0.5*repmat(1+g,1,M).*fliplr(cumprod([ones(NN,1),PopDec(:,1:M-1)],2)).*[ones(NN,1),1-PopDec(:,M-1:-1:1)];
        case 'DTLZ2'
            g      = sum((PopDec(:,M:end)-0.5).^2,2);
            PopObj = repmat(1+g,1,M).*fliplr(cumprod([ones(size(g,1),1),cos(PopDec(:,1:M-1)*pi/2)],2)).*[ones(size(g,1),1),sin(PopDec(:,M-1:-1:1)*pi/2)];
        case 'DTLZ3'
            g      = 100*(D-M+1+sum((PopDec(:,M:end)-0.5).^2-cos(20.*pi.*(PopDec(:,M:end)-0.5)),2));
            PopObj = repmat(1+g,1,M).*fliplr(cumprod([ones(NN,1),cos(PopDec(:,1:M-1)*pi/2)],2)).*[ones(NN,1),sin(PopDec(:,M-1:-1:1)*pi/2)];
            
             

    end
end