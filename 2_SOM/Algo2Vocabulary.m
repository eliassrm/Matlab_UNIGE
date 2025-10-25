%% Properties of clusters (Vocabulary)
clc
clear all
close all

follower = true;

%% load the output of SOM (clustered data)

% load the output of SOM (clustered data)
if follower == true
    load('VocabularySOMF.mat')
else
    load('VocabularySOMA.mat')
end

DataLength        = size(net.data,1) ;
% Assignment of each point to a cluster
nodesInTime       = net.Discrete_data;
% Cell array containing the datapoints assigned to each cluster
datanodes         = net.datanodes ;
% Number of clusters
N                 = net.N;
trajectorySize    = net.UMKF.TrajectorySize;
TotalTrajectories = net.UMKF.num_trajectories;

%% Compute the Vocabulary parts
% compute mean
[nodesMean,nodesMean_Pos ,nodesMean_Vel ] = GetMean(datanodes);
% compute covariance
[nodesCov ,nodesCov_Pos ,nodesCov_Vel ] = GetCovariance(datanodes);
% compute radius (not used in MJPF any more)
[nodesRadAccept, nodesRadAccept_Pos,nodesRadAccept_Vel ] = GetRadius(datanodes);
% compute transition matrix
[transitionMat, DiscreteTrajectories] = GetTransitionMatrix(N,nodesInTime, ...
    trajectorySize,TotalTrajectories);
% compute time transition matrices
timeMats = GetTemporalTimeMat(N,nodesInTime);
% Find the max time spent in each cluster
net = CalculateMaxClustersTime (net);

%% store vocabulary(properties of clusters)
% mean
net.nodesMean = nodesMean;
net.nodesMean_Pos = nodesMean_Pos;
net.nodesMean_Vel = nodesMean_Vel;
% covariance
net.nodesCov = nodesCov;
net.nodesCov_Pos = nodesCov_Pos;
net.nodesCov_Vel = nodesCov_Vel;
% radius
net.nodesRadAccept = nodesRadAccept;
net.nodesRadAccept_Pos = nodesRadAccept_Pos;
net.nodesRadAccept_Vel = nodesRadAccept_Vel;
% transition matrix
net.transitionMat = transitionMat;
% temporal tranition matrices
net.transMatsTime = timeMats;
net.DiscreteTrajectories = DiscreteTrajectories;

if follower == true
    save('VocabularySOMF.mat','net')
else
    save('VocabularySOMA.mat','net')
end

%% plots
mycolors = colorcube;
radius = nodesRadAccept;

h = figure;
hold on
scatter(net.data(:,1),net.data(:,2),60,mycolors(nodesInTime,:),'.','LineWidth',1)    % colored input data
scatter(nodesMean(:,1),nodesMean(:,2),250,'+','k','linewidth',2)                     % for the '+' at mean position of nodes
quiver(nodesMean(:,1),nodesMean(:,2),nodesMean(:,3),nodesMean(:,4),'LineWidth',1.8,'Color','r','AutoScale','on', 'AutoScaleFactor', 0.4)
hold on
% for  numbering of nodes
a = [1:N]'; b = num2str(a); c = cellstr(b);
text(net.w(:,1), net.w(:,2), c,'linewidth',4);
hold on
grid on


