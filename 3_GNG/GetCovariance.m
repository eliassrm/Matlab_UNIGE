
function [nodesCov, nodesCov_Pos, nodesCov_Vel] = GetCovariance(datanodes)
for i = 1:size(datanodes,2)
    %   Calculation of covariance values
    if isempty(datanodes{1,i})
        nodesCov{1,i}     = zeros(size(datanodes{1,1},2),size(datanodes{1,1},2));
        nodesCov_Pos{1,i} = zeros(size(datanodes{1,1},2)/2,size(datanodes{1,1},2)/2);
        nodesCov_Vel{1,i} = zeros(size(datanodes{1,1},2)/2,size(datanodes{1,1},2)/2);
    else
        nodesCov{1,i} = cov(datanodes{1,i},1);
        nodesCov_Pos{1,i} = cov(datanodes{1,i}(1,1:2),1);
        nodesCov_Vel{1,i} = cov(datanodes{1,i}(1,3:4),1);
    end
end
end

