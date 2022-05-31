load('Finalmat213.mat')
for i=1:2088
for j=1:163
    value=Finalmat211(i,j);
    for k=i+1:2088
        pot=value-Finalmat211(k,1);
        if(pot==0)
        Finalmat211(k,:)=0;
        
        
        end
    end
end

end
fprintf('done\n');
fprintf('gone\n');