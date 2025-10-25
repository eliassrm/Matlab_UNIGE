
%% Markov Jump Particle Filter function

function [outputArg1] = MJPF(TestingData, TrainingData, Vocabulary, N)

%% Input:
% TestingData: the testing dataSet
% Vocabulary consists of :
% - Superstates
% - Mean and Covariance of each Superstate
% - Transition Matrix, including temporal ones

transitionMatrix               = Vocabulary.transitionMat;
temporalTransitionMatrix       = Vocabulary.transMatsTime;
meanOfSuperStates              = Vocabulary.nodesMean;
covarianceOfSuperStates        = Vocabulary.nodesCov;
maxClustersTime                = Vocabulary.maxClustersTime;

%% Parameters Used in the Filtering Process:
% totalTestingTime: Total Testing time.
% GSVDimension: # of states in the Generalized State Vector (GSV).
% A: Transition matrix for the continous linear system.
% B: Control input.
% H: Measurement model.
% totNumOfSuperStates: Total Number of SuperStates.
% N: Total number of Particles:

totNumOfSuperStates = size(transitionMatrix,2);
totalTestingTime = size(TestingData, 2);
GSVDimension     = size(TestingData, 1);
A = [eye(GSVDimension/2),zeros(GSVDimension/2); ...
     zeros(GSVDimension/2),zeros(GSVDimension/2)];
B = [eye(GSVDimension/2);eye(GSVDimension/2)];
H = [eye(GSVDimension/2),zeros(GSVDimension/2); ...
     zeros(GSVDimension/2), eye(GSVDimension/2)];

%% Generate Observation Noise (Observation Noise): 
%  v ~ N(0,R) meaning v is gaussian noise with covariance R

% Observation Noise variance
Var_ONoise = 1;  
% Observation Noise mean
Mu_ONoise  = 0;                    
% Standard deviation of the observation noise
Std_ONoise = sqrt(Var_ONoise)';    

%% Initialization of values to save:
predicted_superstate = zeros(N,totalTestingTime);
predicted_state      = zeros(GSVDimension, totalTestingTime, N);
predicted_cov_state  = zeros(GSVDimension, GSVDimension, totalTestingTime, N);
updated_state        = zeros(GSVDimension, totalTestingTime, N);
updated_cov_state    = zeros(GSVDimension, GSVDimension, totalTestingTime, N);

w = zeros(1,N);

beginTimeForPlot = 2;

%% Procedure MJPF
for i = 1:1:totalTestingTime
    
    % Gaussian Observation Noise
    ONoise = Std_ONoise * randn(GSVDimension,totalTestingTime) + ...
             Mu_ONoise*ones(GSVDimension,totalTestingTime);  
    % Observation Noise Covariance Matrix
    R = cov(ONoise');                  

    % ------ INITIAL STEP ------ > %
    if i == 1 % Intiall Step (initial guess of X then estimate S)
        for n = 1:1:N
            predicted_state(:,i,n) = mvnrnd(TestingData(:,1),R)';
            predicted_cov_state_initial = R;
            t(n) = 1;
            weightscoeff(n,i) = 1/N;
            
            %% Observe first Measurement Zt
            current_measurement = TestingData(:,i)';
            
            %% Calculate Message Back-ward Propagated towards discrete-level (S)
            if n == 1
                probability_lamdaS(i,:) = calculateLamdaS(totNumOfSuperStates, current_measurement, ...
                    meanOfSuperStates, R, covarianceOfSuperStates);
            end
            
            probabilita = makedist('Multinomial','Probabilities',probability_lamdaS(i,:));
            predicted_superstate(n,i) = probabilita.random(1,1);
            
            %% UPDATE STEP
            % -- update states -- %
            %% during the update kalman computes the posterior p(x[k] | z[1:k]) = N(x[k] | m[k], P[k])
            [updated_state(:,i,n), updated_cov_state(:,:,i,n)] =...
                kf_update(predicted_state(:,i,n), predicted_cov_state_initial, current_measurement', H, R);
            
            %% Calculate Abnormalities
            % -- continuous level -- %
            % measure bhattacharrya distance between p(xk/xk-1) and p(zk/xk)
            CLA(n,i) = bhattacharyyadistance(predicted_state(:,i,n)',...        
                current_measurement, predicted_cov_state_initial, R);
            
            % measure bhattacharrya distance between p(xk/xk-1) and p(xk/sk)
            CLB(n,i) = bhattacharyyadistance(predicted_state(:,i,n)',...
                meanOfSuperStates(predicted_superstate(n,i),:),predicted_cov_state_initial,...     
                covarianceOfSuperStates{1,predicted_superstate(n,i)});
            
            % -- update weights -- %
            w(n) = weightscoeff(n,i)*probability_lamdaS(i,predicted_superstate(n,i));
        end % end Particles
        
        %% Calculate Histogram before update
        for ii = 1:totNumOfSuperStates
            elements = find(ii == predicted_superstate(:,i));
            histogram_before_update(ii,i) = length(elements);
        end
        
        %% PF Resampling
        w = w/sum(w); % normalize weights
        pd = makedist('Multinomial','Probabilities',w);                % multinomial distribution to pick multiple likely particles
        swap_index = pd.random(1,N);%take N random numbers
        for n = 1:N
            predicted_state_resampled(:,i,n) = predicted_state(:,i,swap_index(n));
            predicted_superstate_resampled(n,i) = predicted_superstate(swap_index(n),i);
            updated_state_resampled(:,i,n) = updated_state(:,i,swap_index(n));
            updated_cov_state_resampled(:,:,i,n) = updated_cov_state(:,:,i,swap_index(n));
            CLA_resampled(n,i) = CLA(swap_index(n), i);
            CLB_resampled(n,i) = CLB(swap_index(n), i);
        end
        predicted_state = predicted_state_resampled;
        predicted_superstate = predicted_superstate_resampled;
        updated_state = updated_state_resampled;
        updated_cov_state = updated_cov_state_resampled;
        CLA = CLA_resampled;
        CLB = CLB_resampled;
        %% Calculate Histogram after update
        for ii = 1:totNumOfSuperStates
            elements = find(ii == predicted_superstate(:,i));
            histogram_after_update(ii,i) = length(elements);
        end
        
        weightscoeff(:,i+1) = 1/N;
        
        %% Calculate Abnormalities
        % -- discrete level -- %
        KLDA = KLD_Abnormality(totNumOfSuperStates, N, histogram_after_update(:, i)', transitionMatrix, probability_lamdaS(i,:));
        sommaKLD_simmetrica(1,i) = KLDA;
        
        %% Calculate Generalized Errors
        [~, indexMaxLamdaS] = max(probability_lamdaS(i,:));
        discreteEvents_basedOn_LamdaS(i,1) = indexMaxLamdaS;
    end
    % ------ end of INITIAL STEP ------ < %
    
    if i > 1
        for n = 1:1:N
            %% Discrete-Level prediction
            % Select row of transition matrix
            transitionMatRow = transitionMatrix(predicted_superstate(n,i-1),:);
            % Considering time matrices, if we have been in a cluster for
            % more than one time instant
            
            maxTimeCurrentCluster = maxClustersTime(predicted_superstate(n,i-1));
            
            if t(n) > 1 && t(n) <= maxTimeCurrentCluster
                % select the temporal transition matrix related to being
                % in the current cluster for t(n) instances
                curr_temporalTransitionMatrix = temporalTransitionMatrix{1, t(n)};
                temporalTransitionMatRow = curr_temporalTransitionMatrix(predicted_superstate(n,i-1),:);
                finalTransitionMatRow = (temporalTransitionMatRow + transitionMatRow)/2;
                finalTransitionMatRow = finalTransitionMatRow/sum(finalTransitionMatRow);
                
            elseif t(n) > 1 && t(n) > maxTimeCurrentCluster
                % select the last temporal transition matrix
                curr_temporalTransitionMatrix = temporalTransitionMatrix{1, maxTimeCurrentCluster};
                temporalTransitionMatRow = curr_temporalTransitionMatrix(predicted_superstate(n,i-1),:);
                
                % If we have spent more time in the cluster than usual, the
                % probability of all clusters becomes more equal
                % This is to avoid getting stuck in a cluster
                probability_passage_to_all = 1*abs(maxTimeCurrentCluster- t(n))/(N*maxTimeCurrentCluster/2);
                
                finalTransitionMatRow = (temporalTransitionMatRow + transitionMatRow)/2 + ...
                                         probability_passage_to_all;    
                finalTransitionMatRow = finalTransitionMatRow/sum(finalTransitionMatRow);
            else
                finalTransitionMatRow = transitionMatRow;
            end
            
            % I find probability of next superstate
            probability = makedist('Multinomial','Probabilities',finalTransitionMatRow/sum(finalTransitionMatRow));
            predicted_superstate(n,i) = probability.random(1,1);
            
            if(predicted_superstate(n,i-1) == predicted_superstate(n,i))
                % If same superstate, add 1
                t(n) = t(n) + 1;                                           
            else
                % Else rinizialize by 1
                t(n) = 1;                                                  
            end
            
            %% Calculate Histogram before update
            for ii = 1:totNumOfSuperStates
                elements = find(ii == predicted_superstate(:,i));
                histogram_before_update(ii,i) = length(elements);
            end
            
            %% Continuous-Level prediction
            % Xt = AXt-1 + BUst-1 + wt
            currentState = updated_state(:,i-1,n);
            currentCov = updated_cov_state(:,:,i-1,n);
            
            U  = meanOfSuperStates(predicted_superstate(n,i-1),(GSVDimension/2)+1:GSVDimension)';
            Q2 = covarianceOfSuperStates{1, predicted_superstate(n,i-1)};
            
            [predicted_state(:,i,n), predicted_cov_state(:,:,i,n)] =...
                kf_predict(currentState, currentCov, A, Q2, B, U);
            
            %% Receive new Measurement Zt
            current_measurement = TestingData(:,i)';
            
            %% Calculate Message Back-ward Propagated towards discrete-level (S)
            if n == 1
                probability_lamdaS(i,:) = calculateLamdaS_basedOn_Maha(...
                    totNumOfSuperStates, current_measurement, ...
                    meanOfSuperStates, R, covarianceOfSuperStates);
            end
            
            %% Calculate Abnormalities
            % -- discrete level -- %
            KLDA = KLD_Abnormality(totNumOfSuperStates, N, histogram_after_update(:, i-1)', ...
                transitionMatrix, probability_lamdaS(i,:));
            sommaKLD_simmetrica(1,i) = KLDA;
            
            % -- continuous level -- %
            % measure bhattacharrya distance between p(xk/xk-1) and p(zk/xk)
            CLA(n,i) = bhattacharyyadistance(predicted_state(:,i,n)',...    
                current_measurement, real(diag(topdm(predicted_cov_state(:,:,i,n)))).*eye(GSVDimension),R);
            
            % ATTENTION: in the CLB we must compare between the
            % predicted state and the mean of the PREVIOUS (t-1) predicted
            % superstate
            % measure bhattacharrya distance between p(xk/xk-1) and p(xk/sk)
            CLB(n,i) = bhattacharyyadistance(predicted_state(:,i,n)',...
                meanOfSuperStates(predicted_superstate(n,i-1),:), ...
                real(diag(topdm(predicted_cov_state(:,:,i,n)))).*eye(GSVDimension),...     
                real(diag(topdm(covarianceOfSuperStates{1,predicted_superstate(n,i-1)}))).*eye(GSVDimension));
            
            %% UPDATE STEP
            % -- update states -- %
            %% during the update kalman computes the posterior p(x[k] | z[1:k]) = N(x[k] | m[k], P[k])
            [updated_state(:,i,n), updated_cov_state(:,:,i,n)] =...
                kf_update(predicted_state(:,i,n), (predicted_cov_state(:,:,i,n)), current_measurement', H, (R));
            
            % -- update superstates -- %
            w(n) = weightscoeff(n,i)*probability_lamdaS(i,predicted_superstate(n,i));
        end
        %% Resampling PF
        w = w/sum(w); % normalize weights
        pd = makedist('Multinomial','Probabilities',w);  
        % Multinomial distribution to pick multiple likely particles
        swap_index = pd.random(1,N);%take N random numbers
        for n = 1:N
            predicted_state_resampled(:,i,n)       = predicted_state(:,i,swap_index(n));
            predicted_cov_state_resampled(:,:,i,n) = predicted_cov_state(:,:,i,swap_index(n));
            predicted_superstate_resampled(n,i)    = predicted_superstate(swap_index(n),i);
            updated_state_resampled(:,i,n)         = updated_state(:,i,swap_index(n));
            updated_cov_state_resampled(:,:,i,n)   = updated_cov_state(:,:,i,swap_index(n));
            CLA_resampled(n,i) = CLA(swap_index(n), i);
            CLB_resampled(n,i) = CLB(swap_index(n), i);
        end
        predicted_state = predicted_state_resampled;
        predicted_cov_state = predicted_cov_state_resampled;
        predicted_superstate = predicted_superstate_resampled;
        updated_state = updated_state_resampled;
        updated_cov_state = updated_cov_state_resampled;
        CLA = CLA_resampled;
        CLB = CLB_resampled;
        %% Calculate Histogram after update
        for ii = 1:totNumOfSuperStates
            elements = find(ii == predicted_superstate(:,i));
            histogram_after_update(ii,i) = length(elements);
        end
        weightscoeff(:,i+1) = 1/N;
    end
    
    %% Saving all the elements of this run
    innovations = predicted_state(:,i,:) - updated_state(:,i,:);
    EstimationAbn.mean_error(1,i) = mean(mean(abs(innovations(1:GSVDimension/2,:,:))));
    [~, minCLA] = min(CLA(:,i));
    EstimationAbn.CLA(1,i) = CLA(minCLA, i);
    [~, minCLB] = min(CLB(:,i));
    EstimationAbn.CLB(1,i) = CLB(minCLB, i);
    EstimationAbn.sommaKLD_simmetrica(1,i) = sommaKLD_simmetrica(1,i);
    EstimationAbn.predicted_superstate = predicted_superstate;
    EstimationAbn.winning_nodes(i) = predicted_superstate(minCLB, i);
    EstimationAbn.lambdas(i, :) = probability_lamdaS(i);
    [max_value, cluster_lambda_max ] = max(probability_lamdaS(i,:));
    EstimationAbn.clusters_lambda_max(i) = cluster_lambda_max;
    
    if i == 1
        
        subplot(6,1,[1,2]);
        scatter(TrainingData(1,:), TrainingData(2,:),'r');
        hold on
        title('Figure 1: Trajectory')
        axis([-20 30 -15 40])
        
    elseif i > beginTimeForPlot

        %% Plotting
        h = figure(1);
        h.Position =[544 63 787 898];
        
        % Trajectory
        subplot(6,1,[1,2]);
        hold on
        scatter(TestingData(1,i), TestingData(2,i),'b', 'filled')
    
        % CLA
    	subplot(6,1,3);
        cla
        plot(EstimationAbn.CLA(beginTimeForPlot:i),'-r')
        title('CLA')
        
        % CLB
        subplot(6,1,4);
        cla
        plot(EstimationAbn.CLB(beginTimeForPlot:i),'-r')
        title('CLB')
        axis([1 totalTestingTime 0 20])

        % KLDA
        subplot(6,1,5);
        cla
        plot(sommaKLD_simmetrica(beginTimeForPlot:i),'-r')
        title('KLDA')

        % Innovation
        subplot(6,1,6);
        cla
        plot(EstimationAbn.mean_error(beginTimeForPlot:i),'-r')
        title('Mean Error')
        
    end
    
end % End for testingData Z

%% Saving all predicted and updated values
EstimationAbn.TrajPred   = predicted_state;
EstimationAbn.TrajUpdate = updated_state;
    
%% Output:
outputArg1 = EstimationAbn;
end
