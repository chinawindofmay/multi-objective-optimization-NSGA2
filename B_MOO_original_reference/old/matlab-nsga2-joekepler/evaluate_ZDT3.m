function f = evaluate_ZDT3( x, M, V )
%EVALUATE_ZDT3 此处显示有关此函数的摘要
%   此处显示详细说明

f = [];
f(1) = x(1);
g = 1;
sum = 0;
for i = 2:V
    sum = sum + x(i);
end
sum = 9*(sum / (V-1));
g = g + sum;
f(2) = g * (1 - sqrt(x(1) / g)-(x(1)/g)*sin(10*pi*x(1)));
end

