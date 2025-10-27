%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
clc 
close all

% addpath('./dataset')

file_training = 'Training_Data_Saffarmoghadam';
file_testing  = 'Testing_Data_Saffarmoghadam.mat';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training
load(file_training)
figure
scatter(PosNoiseX1, PosNoiseY1, 'b')
hold on
scatter(TrajectAttrX1, TrajectAttrY1, 'r')
title('Training dataset')


% Testing
load(file_testing)
figure
scatter(PosNoiseX1, PosNoiseY1, 'b')
hold on
%scatter(PosNoiseX1(:,60), PosNoiseY1(:,60), 'b')
scatter(TrajectAttrX1, TrajectAttrY1, 'r')
title('Testing dataset')

