function chromo_parent = tournament_selection( chromo )
%二进制竞标赛
[pop, suoyin] = size(chromo);
touranment=2;
a=round(pop/2);
chromo_candidate=zeros(touranment,1);
chromo_rank=zeros(touranment,1);
chromo_distance=zeros(touranment,1);
chromo_parent=zeros(a,suoyin);
% 获取等级的索引
rank = suoyin - 1;
% 获得拥挤度的索引
distance = suoyin;
for i=1:a
    for j=1:touranment
        chromo_candidate(j)=round(pop*rand(1));%随机产生候选者
        if(chromo_candidate(j)==0)%索引从1开始
            chromo_candidate(j)=1;
        end
        if(j>1)
            while (~isempty(find(chromo_candidate(1:j-1)==chromo_candidate(j))))
                chromo_candidate(j)=round(pop*rand(1));
                if(chromo_candidate(j)==0)%索引从1开始
                    chromo_candidate(j)=1;
                end
            end
        end
    end
    for j=1:touranment
        chromo_rank(j)=chromo(chromo_candidate(j),rank);
        chromo_distance(j)=chromo(chromo_candidate(j),distance);
    end
    %取出低等级的个体索引
    minchromo_candidate=find(chromo_rank==min(chromo_rank));
    %多个索引按拥挤度排序
    if (length(minchromo_candidate)~=1)
        maxchromo_candidate=find(chromo_distance(minchromo_candidate)==max(chromo_distance(minchromo_candidate)));
        if(length(maxchromo_candidate)~=1)
            maxchromo_candidate = maxchromo_candidate(1);
        end
        chromo_parent(i,:)=chromo(chromo_candidate(minchromo_candidate(maxchromo_candidate)),:);
    else
        chromo_parent(i,:)=chromo(chromo_candidate(minchromo_candidate(1)),:);
    end
end
end