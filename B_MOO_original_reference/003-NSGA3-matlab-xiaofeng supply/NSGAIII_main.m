clc,clear
global N M D  PopCon name gen

N = 10;                        % 种群个数
M = 8;                          % 目标个数
name = 'DTLZ1';                 % 测试函数选择，目前只有：DTLZ1、DTLZ2、DTLZ3
gen = 500;                      %迭代次数
%% Generate the reference points and random population
[Z,N] = UniformPoint(N,M);        % 生成一致性参考解
[res,Population,PF] = funfun(); % 生成初始种群与目标值
Pop_objs = CalObj(Population); % 计算适应度函数值
Zmin  = min(Pop_objs(all(PopCon<=0,2),:),[],1); %求理想点，其实PopCon是处理有约束问题的，这里并没有用到

%% Optimization
for i = 1:gen
    MatingPool = TournamentSelection(2,N,sum(max(0,PopCon),2));
%      test value
    MatingPool=[6, 3, 7, 8, 4, 2, 5, 2]
    Offspring  = GA(Population(MatingPool,:));
    Offspring_objs = CalObj(Offspring);
    Zmin       = min([Zmin;Offspring_objs],[],1);
    Population = EnvironmentalSelection([Population;Offspring],N,Z,Zmin);
    Popobj = CalObj(Population);
    if(M<=3)
        plot3(Popobj(:,1),Popobj(:,2),Popobj(:,3),'ro')
        title(num2str(i));
        drawnow
    end
end

% test value
Popobj=[0.001740539,0.000505777,1.220228085,0.08365739,0.979644224,26.99090338,32.33749364,154.5825837;
0.330216274,2.028785642,0,8.643557572,0.943363137,2.223168159,82.66332125,37.56273832;
0.488432283,0.187703277,0,0.054875526,0.087518438,19.20889645,198.0122316,0;
0,0,26.31358908,2.852262931,19.4835994,9.05379306,78.6376131,0;
0.306789973,1.857447174,7.533014447,0.915552994,36.73179559,8.810956753,78.2768302,0;
1.586282531,7.488620221,0,0.984357766,1.35194207,2.12365533,121.0149938,0;
1.138863793,7.973089515,0,0.982198583,1.356631393,2.131021403,121.4347442,0;
0,0,1.863182801,0.238799603,1.313066638,79.25006072,112.6554837,0]

if(M<=3)
    hold on
    plot3(PF(:,1),PF(:,2),PF(:,3),'g*')
else
    for i=1:size(Popobj,1)
        plot(Popobj(i,:))
        hold on
    end
end
%%IGD
score = IGD(Popobj,PF)