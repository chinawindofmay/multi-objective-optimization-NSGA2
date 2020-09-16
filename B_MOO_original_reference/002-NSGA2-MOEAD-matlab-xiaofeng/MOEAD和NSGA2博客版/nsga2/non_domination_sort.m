function [F,chromo] = non_domination_sort( pop,chromo,f_num,x_num )
%non_domination_sort ��ʼ��Ⱥ�ķ�֧������ͼ���ӵ����
%��ʼ��pareto�ȼ�Ϊ1
pareto_rank=1;
F(pareto_rank).ss=[];%pareto�ȼ�Ϊpareto_rank�ļ���
p=[];%ÿ������p�ļ���
for i=1:pop
    %%%�������Ⱥ��ÿ������p�ı�֧�����n�͸ø���֧��Ľ�ļ���s
    p(i).n=0;%��֧�������Ŀn
    p(i).s=[];%֧���ļ���s
    for j=1:pop
        less=0;%y'��Ŀ�꺯��ֵС�ڸ����Ŀ�꺯��ֵ��Ŀ
        equal=0;%y'��Ŀ�꺯��ֵ���ڸ����Ŀ�꺯��ֵ��Ŀ
        greater=0;%y'��Ŀ�꺯��ֵ���ڸ����Ŀ�꺯��ֵ��Ŀ
        for k=1:f_num
            if(chromo(i,x_num+k)<chromo(j,x_num+k))
                less=less+1;
            elseif(chromo(i,x_num+k)==chromo(j,x_num+k))
                equal=equal+1;
            else
                greater=greater+1;
            end
        end
        if(less==0 && equal~=f_num)%if(less==0 && greater~=0)
            p(i).n=p(i).n+1;%��֧�������Ŀn+1
        elseif(greater==0 && equal~=f_num)%elseif(greater==0 && less~=0)
            p(i).s=[p(i).s j];
        end
    end
    %%%����Ⱥ�в���Ϊn�ĸ�����뼯��F(1)��
    if(p(i).n==0)
        chromo(i,f_num+x_num+1)=1;%�������ĵȼ���Ϣ
        F(pareto_rank).ss=[F(pareto_rank).ss i];
    end
end
%%%��pareto�ȼ�Ϊpareto_rank+1�ĸ���
while ~isempty(F(pareto_rank).ss)
    temp=[];
    for i=1:length(F(pareto_rank).ss)
        if ~isempty(p(F(pareto_rank).ss(i)).s)
            for j=1:length(p(F(pareto_rank).ss(i)).s)
                p(p(F(pareto_rank).ss(i)).s(j)).n=p(p(F(pareto_rank).ss(i)).s(j)).n - 1;%nl=nl-1
                if p(p(F(pareto_rank).ss(i)).s(j)).n==0
                    chromo(p(F(pareto_rank).ss(i)).s(j),f_num+x_num+1)=pareto_rank+1;%�������ĵȼ���Ϣ
                    temp=[temp p(F(pareto_rank).ss(i)).s(j)];
                end
            end
        end
    end
    pareto_rank=pareto_rank+1;
    F(pareto_rank).ss=temp;
end