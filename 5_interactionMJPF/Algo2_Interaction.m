
% MARKOV JUMP PARITCLE FILTER AND ABNORMALITY DETECTION, IN INTERACTIONS
% CASE.

addpath('./MJPF_functions')
addpath('./Additional_codes');

clc
clear
close all

% Testing or training or on testing data
trainingData = false;
% Selected trajectory of MJPF
track = 10; 
% Number of particles of MJPF
N = 100;

%% LOAD INPUT AND DEFINE INDEX OF TRAINING AND TESTING                                                          
%%   Vocabulary information of entire system that favors object 1 (follower) dynamics 
load(['vocabularyGen',num2str(1),'.mat'])         
Vocabulary1 = vocabularyGen;

%%   Vocabulary information of entire system that favors object 2 (attractor) dynamics 
load(['vocabularyGen',num2str(2),'.mat'])                 
Vocabulary2 = vocabularyGen;

%% Definition of the employed data
%   Number of data for each trajectory of test
if trainingData == true
    load('numberData.mat')  
    %   Data test object 1
    load('pos1.mat')      
    %   Data test object 2
    load('pos2.mat')     
    %   Concatenate
    data     = [pos1 pos2];
else
    load('numberDataTest.mat')  
    numberData = numberDataTest;
    %   Data test object 1
    load('pos1Test.mat')      
    %   Data test object 2
    load('pos2Test.mat')   
    %   Concatenate
    data     = [pos1Test pos2Test];
end

dataTest = data';
startInd = sum(numberData(1:track-1,1)) + 1;
endInd   = sum(numberData(1:track-1,1)) + numberData(track);
dataTest = dataTest(:,startInd:endInd);

%% MJPF
EstimationAbn = MJPF_interactions(dataTest, Vocabulary1, Vocabulary2, N);
