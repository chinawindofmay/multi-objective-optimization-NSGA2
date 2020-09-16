function f = object_fun( x,f_num,x_num,fun )
%   ≤‚ ‘∫Ø ˝µƒ…Ë÷√
%--------------------ZDT1--------------------
if strcmp(fun,'ZDT1')
    f=[];
    f(1)=x(1);
    sum=0;
    for i=2:x_num
        sum = sum+x(i);
    end
    g=1+9*(sum/(x_num-1));
    f(2)=g*(1-(f(1)/g)^0.5);
end
%--------------------ZDT2--------------------
if strcmp(fun,'ZDT2')
    f=[];
    f(1)=x(1);
    sum=0;
    for i=2:x_num
        sum = sum+x(i);
    end
    g=1+9*(sum/(x_num-1));
    f(2)=g*(1-(f(1)/g)^2);
end
%--------------------ZDT3--------------------
if strcmp(fun,'ZDT3')
    f=[];
    f(1)=x(1);
    sum=0;
    for i=2:x_num
        sum = sum+x(i);
    end
    g=1+9*(sum/(x_num-1));
    f(2)=g*(1-(f(1)/g)^0.5-(f(1)/g)*sin(10*pi*f(1)));
end
%--------------------ZDT4--------------------
if strcmp(fun,'ZDT4')
    f=[];
    f(1)=x(1);
    sum=0;
    for i=2:x_num
        sum = sum+(x(i)^2-10*cos(4*pi*x(i)));
    end
    g=1+9*10+sum;
    f(2)=g*(1-(f(1)/g)^0.5);
end
%--------------------ZDT6--------------------
if strcmp(fun,'ZDT6')
    f=[];
    f(1)=1-(exp(-4*x(1)))*((sin(6*pi*x(1)))^6);
    sum=0;
    for i=2:x_num
        sum = sum+x(i);
    end
    g=1+9*((sum/(x_num-1))^0.25);
    f(2)=g*(1-(f(1)/g)^2);
end
%--------------------------------------------
%--------------------DTLZ1-------------------
if strcmp(fun,'DTLZ1')
    f=[];
    sum=0;
    for i=3:x_num
        sum = sum+((x(i)-0.5)^2-cos(20*pi*(x(i)-0.5)));
    end
    g=100*(x_num-2)+100*sum;
    f(1)=(1+g)*x(1)*x(2);
    f(2)=(1+g)*x(1)*(1-x(2));
    f(3)=(1+g)*(1-x(1));
end
%--------------------------------------------
%--------------------DTLZ2-------------------
if strcmp(fun,'DTLZ2')
    f=[];
    sum=0;
    for i=3:x_num
        sum = sum+(x(i))^2;
    end
    g=sum;
    f(1)=(1+g)*cos(x(1)*pi*0.5)*cos(x(2)*pi*0.5);
    f(2)=(1+g)*cos(x(1)*pi*0.5)*sin(x(2)*pi*0.5);
    f(3)=(1+g)*sin(x(1)*pi*0.5);
end
%--------------------------------------------
end