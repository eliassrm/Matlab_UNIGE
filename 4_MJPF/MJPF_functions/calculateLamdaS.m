function [outputArg1] = calculateLamdaS(totNumOfSuperstates, measurement, ...
                    meanOfSuperstates, R, covarianceOfSuperstates)
%% Input:
% totNumOfSuperstates: total number of Superstates
% measurement: the current observation
% meanOfSuperstates: matrix consisting of the mean value of each Superstate
% covarianceOfSuperstates: cell consisting of the covariance matrix of each Superstate

%% Calculate lambda in terms of battacharyya distance b/w observation & each superstate:
for index_s = 1:totNumOfSuperstates
    lamdaS(1, index_s) = bhattacharyyadistance(measurement, ...
        meanOfSuperstates(index_s,:), R, covarianceOfSuperstates{1,index_s});
    
    if isnan(lamdaS(1, index_s))
        lamdaS(1, index_s) = max(lamdaS*1000);
    end
end

lamdaS = abs(lamdaS);

%% Convert lamda to a discrete probability distribution:
n = 1; % using n can help to make the probability distribution more skewed (for example if n=1 give you [0.6 0.4], n=2 will give you [0.1 0.9])
temp=(1./((lamdaS).^n));
temp(isnan(temp))=10000;
probability_lamdaS = temp/sum(temp);

%% Output:
outputArg1 = probability_lamdaS;

end

