
function [nodesRadAccept, nodesRadAccept_Pos, nodesRadAccept_Vel]  = GetRadius(datanodes)
for i = 1:size(datanodes,2)
    % calculate radius/ boundary of each node
    if isempty(datanodes{1,i})
        nodesRadAccept(1,i)     = 0;
        nodesRadAccept_Pos(1,i) = 0;
        nodesRadAccept_Vel(1,i) = 0;
    else
        nodesRadAccept(1,i)     = sqrt(sum((3*std(datanodes{1,i},1)).^2));
        nodesRadAccept_Pos(1,i) = sqrt(sum((2*std(datanodes{1,i}(:,1:2),1)).^2));
        nodesRadAccept_Vel(1,i) = sqrt(sum((2*std(datanodes{1,i}(:,3:4),1)).^2));
    end
end
end



