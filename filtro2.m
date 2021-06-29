clear; close all; clc;
a = [1,2,3,4];
Na = length(a);
b = [2,3,4,5];
input_sample = 1:10;
Nx = length(input_sample);
yy = filter(b,a,input_sample); %Filtrado IIR con procesamiento por bloque
u = zeros(1,Na);

for n=1:Nx
    x=input_sample(n);
    temp1=0;
    temp2=0;
    y=0;
    for k=2:Na
        temp1=temp1-a(k)*u(k);
        temp2=temp2+b(k)*u(k);
    end
    u(1) = x+temp1;
    y = u(1)*b(1)+temp2;
    for m=Na:-1:2
        u(m)=u(m-1);
    end
    yy
    y
    pause
end