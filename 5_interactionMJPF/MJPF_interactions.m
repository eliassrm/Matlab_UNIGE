
%% Markov Jump Particle Filter function

function [outputArg1] = MJPF_interactions(TestingData, Vocabulary1, Vocabulary2, N)

%% Input:
% TestingData: the testing dataSet
% Vocabulary consists of :
% - Superstates
% - Mean and Covariance of each Superstate
% - Transition Matrix, including temporal ones

transitionMatrix          = Vocabulary2.transitionMatrix;
temporalTransitionMatrix  = Vocabulary2.transMatsTime;
% NodesMean 1
meanOfSuperStates1        = Vocabulary1.averageN;
meanOfSuperStates1        = cell2mat(meanOfSuperStates1);
meanOfSuperStates1        = meanOfSuperStates1(:,1:4);
% NodesMean 2
meanOfSuperStates2        = Vocabulary2.averageN;
meanOfSuperStates2        = cell2mat(meanOfSuperStates2);
meanOfSuperStates2        = meanOfSuperStates2(:,1:4);
% NodesCov 1 
covarianceOfSuperStates1  = Vocabulary1.covarianceN;
for i = 1:size(covarianceOfSuperStates1, 1)
    covarianceOfSuperStates1{i,1} = covarianceOfSuperStates1{i,1}(1:4, 1:4);
end
covarianceOfSuperStates1  = covarianceOfSuperStates1';
% NodesCov 2
covarianceOfSuperStates2  = Vocabulary2.covarianceN;
for i = 1:size(covarianceOfSuperStates2, 1)
    covarianceOfSuperStates2{i,1} = covarianceOfSuperStates2{i,1}(1:4, 1:4);
end
covarianceOfSuperStates2  = covarianceOfSuperStates2';
% Codebook
codebook        = Vocabulary2.label;
codebookNonNull = codebook(not(codebook(:,4)==0),:);

%% Parameters Used in the Filtering Process:
% totalTestingTime: Total Testing time.
% GSVDimension: # of states in the Generalized State Vector (GSV).
% A: Transition matrix for the continous linear system.
% B: Control input.
% H: Measurement model.
% totNumOfSuperStates: Total Number of SuperStates.
% N: Total number of Particles:

totNumOfSuperStates = size(covarianceOfSuperStates1,2);
totalTestingTime    = size(TestingData, 2);
GSVDimension        = size(TestingData, 1)/2;
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
predicted_superstate1 = zeros(N,totalTestingTime);
predicted_state1      = zeros(GSVDimension, totalTestingTime, N);
predicted_cov_state1  = zeros(GSVDimension, GSVDimension, totalTestingTime, N);
updated_state1        = zeros(GSVDimension, totalTestingTime, N);
updated_cov_state1    = zeros(GSVDimension, GSVDimension, totalTestingTime, N);

predicted_superstate2 = zeros(N,totalTestingTime);
predicted_state2      = zeros(GSVDimension, totalTestingTime, N);
predicted_cov_state2  = zeros(GSVDimension, GSVDimension, totalTestingTime, N);
updated_state2        = zeros(GSVDimension, totalTestingTime, N);
updated_cov_state2    = zeros(GSVDimension, GSVDimension, totalTestingTime, N);

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
            predicted_state1(:,i,n)     = mvnrnd(TestingData(1:4,1),R)';
            predicted_state2(:,i,n)     = mvnrnd(TestingData(5:8,1),R)';
            predicted_cov_state_initial = R;
            t(n) = 1;
            weightscoeff(n,i) = 1/N;
            
            %% Observe first Measurement Zt
            current_measurement1 = TestingData(1:4,i)';
            current_measurement2 = TestingData(5:8,i)';
            
            %% Calculate Message Back-ward Propagated towards discrete-level (S)
            if n == 1
                probability_lamdaS1(i,:) = calculateLamdaS(totNumOfSuperStates, ...
                    current_measurement1, meanOfSuperStates1, R, covarianceOfSuperStates1);
                probability_lamdaS2(i,:) = calculateLamdaS(totNumOfSuperStates, ...
                    current_measurement2, meanOfSuperStates2, R, covarianceOfSuperStates2);
            end
            
            probabilita = makedist('Multinomial','Probabilities',probability_lamdaS1(i,:));
            predicted_superstate1(n,i) = probabilita.random(1,1);
            
            probabilita = makedist('Multinomial','Probabilities',probability_lamdaS2(i,:));
            predicted_superstate2(n,i) = probabilita.random(1,1);
            
            %% UPDATE STEP
            % -- update states -- %
            %% during the update kalman computes the posterior p(x[k] | z[1:k]) = N(x[k] | m[k], P[k])
            [updated_state1(:,i,n), updated_cov_state1(:,:,i,n)] = ...
                kf_update(predicted_state1(:,i,n), predicted_cov_state_initial, ...
                current_measurement1', H, R);
            [updated_state2(:,i,n), updated_cov_state2(:,:,i,n)] = ...
                kf_update(predicted_state2(:,i,n), predicted_cov_state_initial, ...
                current_measurement2', H, R);
            
            %% Calculate Abnormalities
            % -- continuous level -- %
            % measure bhattacharrya distance between p(xk/xk-1) and p(zk/xk)
            CLA1(n,i) = bhattacharyyadistance(predicted_state1(:,i,n)',...        
                current_measurement1, predicted_cov_state_initial, R);
            CLA2(n,i) = bhattacharyyadistance(predicted_state2(:,i,n)',...        
                current_measurement2, predicted_cov_state_initial, R);
            
            % measure bhattacharrya distance between p(xk/xk-1) and p(xk/sk)
            CLB1(n,i) = bhattacharyyadistance(predicted_state1(:,i,n)',...
                meanOfSuperStates1(predicted_superstate1(n,i),:),predicted_cov_state_initial,...     
                covarianceOfSuperStates1{1,predicted_superstate1(n,i)});
            CLB2(n,i) = bhattacharyyadistance(predicted_state2(:,i,n)',...
                meanOfSuperStates2(predicted_superstate2(n,i),:),predicted_cov_state_initial,...     
                covarianceOfSuperStates2{1,predicted_superstate2(n,i)});
            
            % -- update weights -- %
            w(n) = weightscoeff(n,i)*probability_lamdaS1(i,predicted_superstate1(n,i)) + ...
                   weightscoeff(n,i)*probability_lamdaS2(i,predicted_superstate2(n,i));

            %% Identify the codebook's label in which we are

            predicted_label(n,i) = FindClosestCodebookLabel(predicted_superstate1(n,i), ...
                predicted_superstate2(n,i), codebook, codebookNonNull, totNumOfSuperStates, ...
                probability_lamdaS1, probability_lamdaS2);
            
        end % end Particles
        
        %% PF Resampling
        w = w/sum(w); % normalize weights
        pd = makedist('Multinomial','Probabilities',w);                % multinomial distribution to pick multiple likely particles
        swap_index = pd.random(1,N);%take N random numbers
        for n = 1:N
            predicted_state_resampled1(:,i,n)     = predicted_state1(:,i,swap_index(n));
            predicted_superstate_resampled1(n,i)  = predicted_superstate1(swap_index(n),i);
            updated_state_resampled1(:,i,n)       = updated_state1(:,i,swap_index(n));
            updated_cov_state_resampled1(:,:,i,n) = updated_cov_state1(:,:,i,swap_index(n));
            CLA_resampled1(n,i)                   = CLA1(swap_index(n), i);
            CLB_resampled1(n,i)                   = CLB1(swap_index(n), i);
            predicted_state_resampled2(:,i,n)     = predicted_state2(:,i,swap_index(n));
            predicted_superstate_resampled2(n,i)  = predicted_superstate2(swap_index(n),i);
            updated_state_resampled2(:,i,n)       = updated_state2(:,i,swap_index(n));
            updated_cov_state_resampled2(:,:,i,n) = updated_cov_state2(:,:,i,swap_index(n));
            CLA_resampled2(n,i)                   = CLA2(swap_index(n), i);
            CLB_resampled2(n,i)                   = CLB2(swap_index(n), i);
        end
        predicted_state1      = predicted_state_resampled1;
        predicted_superstate1 = predicted_superstate_resampled1;
        updated_state1        = updated_state_resampled1;
        updated_cov_state1    = updated_cov_state_resampled1;
        CLA1                  = CLA_resampled1;
        CLB1                  = CLB_resampled1;
        predicted_state2      = predicted_state_resampled2;
        predicted_superstate2 = predicted_superstate_resampled2;
        updated_state2        = updated_state_resampled2;
        updated_cov_state2    = updated_cov_state_resampled2;
        CLA2                  = CLA_resampled2;
        CLB2                  = CLB_resampled2;
        
        %% Calculate Histogram after update
        for ii = 1:totNumOfSuperStates
            elements = find(ii == predicted_superstate1(:,i));
            histogram_after_update1(ii,i) = length(elements);
        end
        for ii = 1:totNumOfSuperStates
            elements = find(ii == predicted_superstate2(:,i));
            histogram_after_update2(ii,i) = length(elements);
        end
        
        weightscoeff(:,i+1) = 1/N;

    end
    % ------ end of INITIAL STEP ------ < %
    
    %% FOLLOWING TIME INSTANTS
    
    if i > 1
        for n = 1:1:N
            
            %% Discrete-Level prediction
            % Select row of transition matrix
            currentLabel     = predicted_label(n,i-1);
            transitionMatRow = transitionMatrix(currentLabel,:);
            % Considering time matrices, if we have been in a cluster for
            % more than one time instant
            
            if t(n) > 1 && t(n) <= size(temporalTransitionMatrix,2)
                % select the temporal transition matrix related to being
                % in the current cluster for t(n) instances
                curr_temporalTransitionMatrix = temporalTransitionMatrix{1, t(n)};
                temporalTransitionMatRow = curr_temporalTransitionMatrix(currentLabel,:);
                finalTransitionMatRow = (temporalTransitionMatRow + transitionMatRow)/2;
                finalTransitionMatRow = finalTransitionMatRow/sum(finalTransitionMatRow);
            elseif t(n) > 1 && t(n) > size(temporalTransitionMatrix,2)
                % reselect the label, if we have spent too much time in a 
                % cluster (to avoid getting stuck)
                
                % Resample the clusters from the two objects based on the
                % lambda vectors
                probability1 = makedist('Multinomial','Probabilities',...
                    real(probability_lamdaS1(i-1,:)/sum(probability_lamdaS1(i-1,:))));
                predicted_superstate1(n,i-1) = probability1.random(1,1);
                probability2 = makedist('Multinomial','Probabilities',...
                    real(probability_lamdaS2(i-1,:)/sum(probability_lamdaS2(i-1,:))));
                predicted_superstate2(n,i-1) = probability2.random(1,1);
                
                % Estimate the label at previous time instantfrom those of the non null
                % codebook
                predicted_label(n,i-1)  = FindClosestCodebookLabel(predicted_superstate1(n,i-1), ...
                    predicted_superstate2(n,i-1), codebook, codebookNonNull, totNumOfSuperStates, ...
                    probability_lamdaS1, probability_lamdaS2);
                
                currentLabel          = predicted_label(n,i-1);
                
                % Take the transition matrix row relative to that label
                finalTransitionMatRow = transitionMatrix(currentLabel,:);
            else
                finalTransitionMatRow = transitionMatRow;
            end
            
            % I find probability of next superstate
            probability = makedist('Multinomial','Probabilities',finalTransitionMatRow);
            predicted_label(n,i) = probability.random(1,1);
            
            % Take superstate from codebook
            predicted_superstate1(n,i) = codebookNonNull(predicted_label(n,i),1);
            predicted_superstate2(n,i) = codebookNonNull(predicted_label(n,i),2);
            
            if(predicted_label(n,i-1) == predicted_label(n,i))
                % If same superstate, add 1
                t(n) = t(n) + 1;                                           
            else
                % Else rinizialize by 1
                t(n) = 1;                                                  
            end
            
            %% Continuous-Level prediction
            % Xt = AXt-1 + BUst-1 + wt
            currentState1 = updated_state1(:,i-1,n);
            currentCov1 = updated_cov_state1(:,:,i-1,n);
            currentState2 = updated_state2(:,i-1,n);
            currentCov2 = updated_cov_state2(:,:,i-1,n);

            U1 = meanOfSuperStates1(predicted_superstate1(n,i-1),(GSVDimension/2)+1:GSVDimension)';
            Q1 = covarianceOfSuperStates1{1, predicted_superstate1(n,i-1)};
            U2 = meanOfSuperStates2(predicted_superstate2(n,i-1),(GSVDimension/2)+1:GSVDimension)';
            Q2 = covarianceOfSuperStates2{1, predicted_superstate2(n,i-1)};
            
            [predicted_state1(:,i,n), predicted_cov_state1(:,:,i,n)] =...
                kf_predict(currentState1, currentCov1, A, Q1, B, U1);
            [predicted_state2(:,i,n), predicted_cov_state2(:,:,i,n)] =...
                kf_predict(currentState2, currentCov2, A, Q2, B, U2);
            
            %% Receive new Measurement Zt
            current_measurement1 = TestingData(1:4,i)';
            current_measurement2 = TestingData(5:8,i)';
            
            %% Calculate Message Back-ward Propagated towards discrete-level (S)
            if n == 1
                probability_lamdaS1(i,:) = calculateLamdaS_basedOn_Maha(...
                    totNumOfSuperStates, current_measurement1, ...
                    meanOfSuperStates1, R, covarianceOfSuperStates1);
                probability_lamdaS2(i,:) = calculateLamdaS_basedOn_Maha(...
                    totNumOfSuperStates, current_measurement2, ...
                    meanOfSuperStates2, R, covarianceOfSuperStates2);
            end
            
            %% Calculate Abnormalities
            
            % -- continuous level -- %
            % measure bhattacharrya distance between p(xk/xk-1) and p(zk/xk)
            CLA1(n,i) = bhattacharyyadistance(predicted_state1(:,i,n)', current_measurement1, ...
                       real(diag(topdm(predicted_cov_state1(:,:,i,n)))).*eye(GSVDimension),R);
            CLA2(n,i) = bhattacharyyadistance(predicted_state2(:,i,n)', current_measurement2, ...
                       real(diag(topdm(predicted_cov_state2(:,:,i,n)))).*eye(GSVDimension),R);
            
            % ATTENTION: in the CLB we must compare between the
            % predicted state and the mean of the PREVIOUS (t-1) predicted
            % superstate
            % measure bhattacharrya distance between p(xk/xk-1) and p(xk/sk)
            CLB1(n,i) = bhattacharyyadistance(predicted_state1(:,i,n)',...
                meanOfSuperStates1(predicted_superstate1(n,i-1),:), ...
                real(diag(topdm(predicted_cov_state1(:,:,i,n)))).*eye(GSVDimension),...     
                real(diag(topdm(covarianceOfSuperStates1{1,predicted_superstate1(n,i-1)}))).*eye(GSVDimension));
            CLB2(n,i) = bhattacharyyadistance(predicted_state2(:,i,n)',...
                meanOfSuperStates2(predicted_superstate2(n,i-1),:), ...
                real(diag(topdm(predicted_cov_state2(:,:,i,n)))).*eye(GSVDimension),...     
                real(diag(topdm(covarianceOfSuperStates2{1,predicted_superstate2(n,i-1)}))).*eye(GSVDimension));
            
            %% UPDATE STEP
            % -- update states -- %
            %% during the update kalman computes the posterior p(x[k] | z[1:k]) = N(x[k] | m[k], P[k])
            [updated_state1(:,i,n), updated_cov_state1(:,:,i,n)] =...
                kf_update(predicted_state1(:,i,n), (predicted_cov_state1(:,:,i,n)), current_measurement1', H, (R));
            [updated_state2(:,i,n), updated_cov_state2(:,:,i,n)] =...
                kf_update(predicted_state2(:,i,n), (predicted_cov_state2(:,:,i,n)), current_measurement2', H, (R));
            
            % -- update superstates -- %
            w(n) = weightscoeff(n,i)*probability_lamdaS1(i,predicted_superstate1(n,i)) + ...
                   weightscoeff(n,i)*probability_lamdaS2(i,predicted_superstate2(n,i));
        end
        %% Resampling PF
        w = w/sum(w); % normalize weights
        pd = makedist('Multinomial','Probabilities',real(w));  
        % Multinomial distribution to pick multiple likely particles
        swap_index = pd.random(1,N);%take N random numbers
        
        for n = 1:N
            predicted_state_resampled1(:,i,n)     = predicted_state1(:,i,swap_index(n));
            predicted_superstate_resampled1(n,i)  = predicted_superstate1(swap_index(n),i);
            predicted_cov_state_resampled1(:,:,i,n) = predicted_cov_state1(:,:,i,swap_index(n));
            updated_state_resampled1(:,i,n)       = updated_state1(:,i,swap_index(n));
            updated_cov_state_resampled1(:,:,i,n) = updated_cov_state1(:,:,i,swap_index(n));
            CLA_resampled1(n,i)                   = CLA1(swap_index(n), i);
            CLB_resampled1(n,i)                   = CLB1(swap_index(n), i);
            predicted_state_resampled2(:,i,n)     = predicted_state2(:,i,swap_index(n));
            predicted_superstate_resampled2(n,i)  = predicted_superstate2(swap_index(n),i);
            predicted_cov_state_resampled2(:,:,i,n) = predicted_cov_state2(:,:,i,swap_index(n));
            updated_state_resampled2(:,i,n)       = updated_state2(:,i,swap_index(n));
            updated_cov_state_resampled2(:,:,i,n) = updated_cov_state2(:,:,i,swap_index(n));
            CLA_resampled2(n,i)                   = CLA2(swap_index(n), i);
            CLB_resampled2(n,i)                   = CLB2(swap_index(n), i);
        end
        predicted_state1      = predicted_state_resampled1;
        predicted_superstate1 = predicted_superstate_resampled1;
        predicted_cov_state1  = predicted_cov_state_resampled1;
        updated_state1        = updated_state_resampled1;
        updated_cov_state1    = updated_cov_state_resampled1;
        CLA1                  = CLA_resampled1;
        CLB1                  = CLB_resampled1;
        predicted_state2      = predicted_state_resampled2;
        predicted_superstate2 = predicted_superstate_resampled2;
        predicted_cov_state2  = predicted_cov_state_resampled2;
        updated_state2        = updated_state_resampled2;
        updated_cov_state2    = updated_cov_state_resampled2;
        CLA2                  = CLA_resampled2;
        CLB2                  = CLB_resampled2;
        
        %% Calculate Histogram after update
        for ii = 1:totNumOfSuperStates
            elements = find(ii == predicted_superstate1(:,i));
            histogram_after_update1(ii,i) = length(elements);
        end
        for ii = 1:totNumOfSuperStates
            elements = find(ii == predicted_superstate2(:,i));
            histogram_after_update2(ii,i) = length(elements);
        end
        weightscoeff(:,i+1) = 1/N;
    end
    
    %% Saving all the elements of this run
    innovations1 = predicted_state1(:,i,:) - updated_state1(:,i,:);
    EstimationAbn.mean_error1(1,i) = mean(mean(abs(innovations1(1:GSVDimension/2,:,:))));
    innovations2 = predicted_state2(:,i,:) - updated_state2(:,i,:);
    EstimationAbn.mean_error2(1,i) = mean(mean(abs(innovations2(1:GSVDimension/2,:,:))));
    
    [~, minCLA1] = min(CLA1(:,i));
    EstimationAbn.CLA1(1,i) = CLA1(minCLA1, i);
    [~, minCLA2] = min(CLA2(:,i));
    EstimationAbn.CLA2(1,i) = CLA2(minCLA2, i);
    
    [~, minCLB1] = min(CLB1(:,i));
    EstimationAbn.CLB1(1,i) = CLB1(minCLB1, i);
    [~, minCLB2] = min(CLB2(:,i));
    EstimationAbn.CLB2(1,i) = CLB2(minCLB2, i);
    
    EstimationAbn.predicted_superstate1 = predicted_superstate1;
    EstimationAbn.predicted_superstate2 = predicted_superstate2;
    
    EstimationAbn.winning_nodes1(i) = predicted_superstate1(minCLB1, i);
    EstimationAbn.winning_nodes2(i) = predicted_superstate2(minCLB2, i);
    
    EstimationAbn.lambdas1(i, :) = probability_lamdaS1(i);
    [max_value1, cluster_lambda_max1 ] = max(probability_lamdaS1(i,:));
    EstimationAbn.lambdas2(i, :) = probability_lamdaS2(i);
    [max_value2, cluster_lambda_max2 ] = max(probability_lamdaS2(i,:));
    
    EstimationAbn.clusters_lambda_max1(i) = cluster_lambda_max1;
    EstimationAbn.clusters_lambda_max2(i) = cluster_lambda_max2;
    
        
    if i > beginTimeForPlot

        %% Plotting
        h = figure(1);
        h.Position =[544 63 787 898];
        
        % Trajectory
        subplot(5,1,[1,2]);
        hold on
        scatter(TestingData(1,i), TestingData(2,i),'r', 'filled')
        hold on
        scatter(TestingData(5,i), TestingData(6,i),'b', 'filled')
        axis([min([TestingData(1,:), TestingData(5,:)]) ...
              max([TestingData(1,:), TestingData(5,:)]) ...
              min([TestingData(2,:), TestingData(6,:)]) ...
              max([TestingData(2,:), TestingData(6,:)])]);
    
        % CLA
    	subplot(5,1,3);
        cla
        plot(EstimationAbn.CLA1(beginTimeForPlot:i),'-r')
        hold on
        plot(EstimationAbn.CLA2(beginTimeForPlot:i),'-b')
        title('CLA')
        axis([1 totalTestingTime 0 5])
        legend({'follower', 'attractor'})
        
        % CLB
        subplot(5,1,4);
        cla
        plot(EstimationAbn.CLB1(beginTimeForPlot:i),'-r')
        hold on
        plot(EstimationAbn.CLB2(beginTimeForPlot:i),'-b')
        title('CLB')
        axis([1 totalTestingTime 0 20])
        legend({'follower', 'attractor'})

        % Innovation
        subplot(5,1,5);
        cla
        plot(EstimationAbn.mean_error1(beginTimeForPlot:i),'-r')
        hold on
        plot(EstimationAbn.mean_error2(beginTimeForPlot:i),'-b')
        title('Mean Error')
        axis([1 totalTestingTime 0 1.5])
        legend({'follower', 'attractor'})
        
    end
    
end % End for testingData Z

%% Saving all predicted and updated values
EstimationAbn.TrajPred1   = predicted_state1;
EstimationAbn.TrajUpdate1 = updated_state1;

EstimationAbn.TrajPred2   = predicted_state2;
EstimationAbn.TrajUpdate2 = updated_state2;
    
%% Output:
outputArg1 = EstimationAbn;
end
