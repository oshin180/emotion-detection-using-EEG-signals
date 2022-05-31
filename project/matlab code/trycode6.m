clear all
clc

load('Dt_EEGFeatures.mat')
load('Finale.mat')
load('Finalmat11.mat')
 i=1;
 j=2;
 Finalmat11=[];


while(i<=32)
    for k=1:310
y=unnamed1(k,i);
z=unnamed1(k,j);
if y~=0
    
       m=EEGFeatures_58{1,y}(z,:);
       Finalmat11=[Finalmat11;m];
    
        
    
end
    end
   
    i=i+2;
    j=j+2;
end
