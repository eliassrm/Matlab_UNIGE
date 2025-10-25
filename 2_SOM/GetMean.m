function [nodesMean, nodesMean_Pos, nodesMean_Vel] = GetMean(datanodes)
nodesMean = [];
nodesMean_Pos = [];
nodesMean_Vel = [];
for i = 1:size(datanodes,2)
    %   Calculation of mean values
    if isempty(datanodes{1,i})
        nodesMean     = [nodesMean; zeros(1,size(datanodes{1,1},2))];
        nodesMean_Pos = [nodesMean_Pos;zeros(1,size(datanodes{1,1},2)/2)];
        nodesMean_Vel = [nodesMean_Vel;zeros(1,size(datanodes{1,1},2)/2)];
    else
        nodesMean     = [nodesMean; mean(datanodes{1,i},1)];
        nodesMean_Pos = [nodesMean_Pos; mean(datanodes{1,i}(:,1:2),1)];
        nodesMean_Vel = [nodesMean_Vel; mean(datanodes{1,i}(:,3:4),1)];
    end
end

end
