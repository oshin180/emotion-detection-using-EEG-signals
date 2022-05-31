clear all
clc

load('Finalmat2.mat')

i=1;
j=2;
valueR=0;
valueC=0;

while i<=2088

    fprintf('%d\t',i);
    
    valueR=(Finalmat2(1,i));
    valueC=(Finalmat2(2,i));
    
    while j<=2088
    totR=(valueR-Finalmat2(1,j));
    totC=(valueC-Finalmat2(2,j));
    if(totR==totC && totR==0)
        
        fprintf('%d\t',j);
        j=j+1;
    else
        j=j+1;
    
    
    end
    end
    
    fprintf('\n');
    i=i+1;
    j=i+1;
    
    
end    