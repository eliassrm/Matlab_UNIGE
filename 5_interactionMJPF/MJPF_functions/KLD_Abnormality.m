function [outputArg1] = KLD_Abnormality(totNumOfSuperstates, N, histogram,...
    transitionMat, probability_lamdaS, KLDAbnMax)

%% Input:
% totNumOfSuperstates: total number of Superstates
% N: total number of Particles
% histogram: histogram at time t-1 (after PF resampling)
% transitionMat: the transition matrix learned from previous experience
% probability_lamdaS: probability vector representing a discrete probability disctribution

%% Procedure:
sommaKLD_simmetrica = 0;
for indKLD = 1:totNumOfSuperstates
    
    % Find the weight of the current cluster in the histogram
    particella = histogram(1,indKLD);
    
    % If the weights is non zero:
    if particella>0
        
        % Take the row of the transition matrix related to the cluster
        PP = transitionMat(indKLD,:)+1e-20; % add 1e-100 since KLD doesnt allow zero values
        % And take the probability of being in each cluster
        QQ = probability_lamdaS;
        
        % (Here consider nan or inf cases)
        if isinf(QQ)
            KLD_simmetrica = KLDAbnMax;
        end
        
        if sum(isnan(QQ)) >= 1
            KLD_simmetrica = KLDAbnMax;
        elseif sum(isnan(PP)) >= 1
            KLD_simmetrica = KLDAbnMax;
        else
            % <------ ACTUAL KLDA calculation
            % KLD between PP and QQ, weighted by the weight of the cluster
            % in the histogram.
            KLD_simmetrica = (particella/N)*KLDiv(PP,QQ) + (particella/N)*KLDiv(QQ,PP); %to achieve symmerty
        end
        
        if isinf(KLD_simmetrica)
            KLD_simmetrica = KLDAbnMax;
        end
        
        % Sum all KLDA values until this point
        sommaKLD_simmetrica = sommaKLD_simmetrica + KLD_simmetrica;
    end
end

%% Output:
outputArg1 = sommaKLD_simmetrica;

end

