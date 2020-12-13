function index = TournamentSelection(K,N,varargin)
%TournamentSelection - Tournament selection.
%
%   P = TournamentSelection(K,N,fitness1,fitness2,...) returns the indices
%   of N solutions by K-tournament selection based on their fitness values.
%   In each selection, the candidate having the MINIMUM fitness1 value will
%   be selected; if more than one candidates have the same minimum value of
%   fitness1, then compare their fitness2 values, and so on.
%
%   Example:
%       P = TournamentSelection(2,100,FrontNo)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    fff=0;
    varargin2 = cellfun(@(S) reshape(S,[],1),varargin,'UniformOutput',false);
%     x = [1 4 3 5; 1 3 2 6]；sortrows(x)其结果是按照row来排列，默认首先排第一列，1和1一样大，那么排第二列，3比4小，所以1 3 2 6应该在第一行。
    [~,rank] = sortrows([varargin2{:}]);
    [~,rank] = sort(rank);
    Parents  = randi(length(varargin2{1}),K,N);
    [~,best] = min(rank(Parents),[],1);
    index    = Parents(best+(0:N-1)*K);
end