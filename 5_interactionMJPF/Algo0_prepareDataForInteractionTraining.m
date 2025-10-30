
clc 
close all
clear all

%% Training or testing data?
trainingData = false;

%% Loading the data
if trainingData == true
    load('dataAttractor.mat')
    dataAttractor = data;
    load('dataFollower.mat')
    dataFollower = data;
else
    load('dataAttractorTest.mat')
    dataAttractor = data;
    load('dataFollowerTest.mat')
    dataFollower = data;
end

%% Create array with number of datapoints for each trajectory
numberData = data.TrajectorySize;

%% Data of follower
pos1 = dataFollower.MM;

%% Data of attractor must be repeated for all trajectories
dataSingleAttractor = dataAttractor.MM;
pos2 = [];
for i = 1:length(numberData)
    lengthOfCurrentTrajectory = numberData(i);
    currentTrajectory = dataSingleAttractor(1:lengthOfCurrentTrajectory,:);
    pos2 = [pos2; currentTrajectory];
end

%% Save data
if trainingData == true
    save('numberData.mat', 'numberData');
    save('pos1.mat', 'pos1');
    save('pos2.mat', 'pos2');
else
    numberDataTest = numberData;
    pos1Test       = pos1;
    pos2Test       = pos2;
    save('numberDataTest.mat', 'numberDataTest');
    save('pos1Test.mat', 'pos1Test');
    save('pos2Test.mat', 'pos2Test');
end
