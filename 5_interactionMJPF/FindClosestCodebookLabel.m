
function [predicted_label] = FindClosestCodebookLabel(predicted_superstate1, predicted_superstate2, ...
    codebook, codebookNonNull, totNumOfSuperStates, probability_lamdaS1, probability_lamdaS2)

% Predicted superstate 1
s1 = predicted_superstate1;
% Predicted superstate 2
s2 = predicted_superstate2;

% Look if this couple has been observed
predicted_label = codebook((s1-1)*...
        totNumOfSuperStates+s2,4); 
% If the label was not present, find the closest one
% based on the cluster observations.
if predicted_label == 0
    sumClusterProbabilities     = probability_lamdaS1(codebookNonNull(:,1)) + ...
                                  probability_lamdaS2(codebookNonNull(:,2));
    [~, selectedLineOfCodebook] = min(sumClusterProbabilities);
    predicted_label        = codebookNonNull(selectedLineOfCodebook,4);
end

end 