
%% Fourth  Algorithm: (Abnormality detection)
close all
clear
clc

%% Set parameters
% True if using training data, false if using testing data
dataTrainingBool  = true;
% True if choosing follower data, false if choosing attractor data
attractor         = false;
% Trajectory index to select
trajectoryNum     = 10; 
% Choosing SOM or GNG
clusteringGNG     = true;

% Number of particles
N = 20;

%% Add MJPF functions
addpath('./MJPF_functions')
set(0,'defaultfigurecolor',[1 1 1])

%% Select clustering
if attractor == true
    if clusteringGNG == true
        load ('VocabularyGNGA.mat');
    else
        load ('VocabularySOMA.mat');
    end
else
    if clusteringGNG == true
        load ('VocabularyGNGF.mat');
    else
        load ('VocabularySOMF.mat');
    end
end

%% Data for testing
if dataTrainingBool == true
    % Attractor training data
    if attractor == true
        load ('dataAttractor.mat');
        testingData  = data.MMCell{1,1};
        trainingData = data.MMCell;
    % Follower training data
    else
        load ('dataFollower.mat');
        testingData  = data.MMCell{trajectoryNum};
        trainingData = data.MMCell;
    end
else
    % Attractor testing data
    if attractor == true
        load('dataAttractorTest.mat')
        testingData  = data.MMCell{1,1};
        load ('dataAttractor.mat');
        trainingData = data.MMCell;
    % Follower testing data
    else
        load ('dataFollowerTest.mat');
        testingData  = data.MMCell{trajectoryNum};
        load ('dataFollower.mat');
        trainingData = data.MMCell;
    end
end
                                      
%% MJPF application
trainingData    = cell2mat(trainingData');
figure
[estimationAbn] = MJPF(testingData', trainingData', net, N);

%% Mean and Covariance (training data)
% Mean neurons of position data
averageState = net.nodesMean_Pos;   
% Mean neurons of velocity data
averageDiv   = net.nodesMean_Vel;                                              
% Acceptance neuron radius of position data
radiusState  = net.nodesRadAccept_Pos;  
% Acceptance neuron radius of velocity data
radiusDiv    = net.nodesRadAccept_Vel;  

%% Final Plotting
% Plot learned trajectory (boundary of clusters, number of clusters, )
t          = figure;
t.Position = [544 100 987 898];
hax        = axes;
% Writing the cluster number
a = [1:net.N]'; b = num2str(a); c = cellstr(b);
text(hax, averageState(:,1)+0.5, averageState(:,2)+0.5, c,'linewidth',4);
hold on
% Plotting the mean Velocity of clusters
quiver(hax, averageState(:,1),averageState(:,2), ...
       averageDiv(:,1)  ,averageDiv(:,2),'b','AutoScale','off');
% Plotting the mean Position of clusters
scatter(hax, averageState(:,1),averageState(:,2),'ob')
% Plotting the circles of radius
for a = 1:net.N
    x = averageState(a,1)- radiusState(a);
    y = averageState(a,2) - radiusState(a);
    posPage = [x y 2*radiusState(a) 2*radiusState(a)];
    r = rectangle(hax, 'Position',posPage,'Curvature',[1 1], 'LineStyle', ':');
end
title('Figure 2: observation vs. prediction vs. update')
for i = 1:size(testingData,1)

    % Plotting observed velocity 
    q = quiver(hax, testingData(i,1), testingData(i,2), ...
               testingData(i,3), testingData(i,4),'r', 'Autoscale','off');
    q.MaxHeadSize = 5;
    % Plotting prediction
    quiver(hax, estimationAbn.TrajPred(1,i,end),estimationAbn.TrajPred(2,i,end), ...
        estimationAbn.TrajPred(3,i,end),estimationAbn.TrajPred(4,i,end),'k','AutoScale','off');
    
    scatter(hax, estimationAbn.TrajPred(1,i,end),estimationAbn.TrajPred(2,i,end),'+k')
    % Plotting update
    quiver(hax, estimationAbn.TrajUpdate(1,i,end),estimationAbn.TrajUpdate(2,i,end), ...
        estimationAbn.TrajUpdate(3,i,end),estimationAbn.TrajUpdate(4,i,end),'g','AutoScale','off');
    scatter(hax, estimationAbn.TrajUpdate(1,i,end),estimationAbn.TrajUpdate(2,i,end),'+g')
    box on
    grid on
    %{
    legend({'Clusters velocity', 'Clusters position', ...
            ['Observed velocity at time ', num2str(i)], ...
            'Predicted velocity', 'Predicted position', ...
            'Updated velocity', 'Updated position'});
    %}
    pause(0.1)
end

%% Save the results
if dataTrainingBool == true 
    save(['abn_train_follower_traj' , num2str(trajectoryNum)], 'estimationAbn');
else
    save(['abn_test_follower_traj' , num2str(trajectoryNum)] , 'estimationAbn');
end
