function f = evaluate_objective(x, M, V)%%计算每个个体的M个目标函数值
f = [];
f(1) = x(1);
g = 1;
sum = 0;
for i = 2:V
    sum = sum + x(i);
end
sum = 9*(sum / (V-1));
g = g + sum;
f(2) = g * (1 - sqrt(x(1) / g));
end