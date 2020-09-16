function [W,N] = UniformPoint(N,M)
%UniformPoint - Generate a set of uniformly distributed points on the unit
%hyperplane.
%
%   [W,N] = UniformPoint(N,M) returns approximately N uniformly distributed
%   points with M objectives on the unit hyperplane.
%
%   Due to the requirement of uniform distribution, the number of points
%   cannot be arbitrary, and the number of points in W may be slightly
%   smaller than the predefined size N.
%
%   Example:
%       [W,N] = UniformPoint(275,10)
    H1 = 1;
    while nchoosek(H1+M-1,M-1) <= N
        H1 = H1 + 1;
    end
    H1=H1 - 1;
    W = nchoosek(1:H1+M-1,M-1) - repmat(0:M-2,nchoosek(H1+M-1,M-1),1) - 1;%nchoosek(v,M),v是一个向量，返回一个矩阵，该矩阵的行由每次从v中的M个元素取出k个取值的所有可能组合构成。比如：v=[1,2,3],n=2,输出[1,2;1,3;2,3]
    W = ([W,zeros(size(W,1),1)+H1]-[zeros(size(W,1),1),W])/H1;
    if H1 < M
        H2 = 0;
        while nchoosek(H1+M-1,M-1)+nchoosek(H2+M-1,M-1) <= N
            H2 = H2 + 1;
        end
        H2 = H2 - 1;
        if H2 > 0
            W2 = nchoosek(1:H2+M-1,M-1) - repmat(0:M-2,nchoosek(H2+M-1,M-1),1) - 1;
            W2 = ([W2,zeros(size(W2,1),1)+H2]-[zeros(size(W2,1),1),W2])/H2;
            W  = [W;W2/2+1/(2*M)];
        end
    end
    W = max(W,1e-6);
    N = size(W,1);
end