
%   VOCABULARY EXTRACTION
%   Application of SOM over trajectories to create vocabulary of dynamics
%   Input: ['Training' num2str(numExp)]: contiunus state space of follower and attractor
%   And the number of each trajectory ['numberData' num2str(numExp)
%   Output: ['vocabularyGen' num2str(numData)'.mat'] and M.mat
%   Notes: this code should be run two times, first time 'object = 1' and
%   Second time 'object = 2'.

clc
clear
close all
curDir = pwd;
set(0,'defaultfigurecolor',[1 1 1])
addpath('./somtoolbox');
addpath('./Additional_codes');

object = 2;

%% LOAD DATA 
% Continuos state space of follower
load('pos1.mat')          
% Continuos state space of attractor
load('pos2.mat')     
% 8-dimensional state
% [Xt Yt Xt_dot Yt_dot Xa Ya Xa_dot Ya_dot] 
% vector of follower and attractor (input of clustering algorithm)
data1 = [pos1 pos2];  
% 8-dimensional state
% [Xa Ya Xa_dot Ya_dot Xt Yt Xt_dot Yt_dot ] 
% vector of attractor and follower (input of clustering algorithm)
data2 = [pos2 pos1];  
% Number of datapoints for each trajectory
load('numberData.mat')    

%% PARAMETERS OF SOM
% Alpha is the weight associated to the 3rd and 4th value inside
% the data array given as input to the clustering algorithm.
% This means that the velocity of the follower (when object = 1)
% and the velocity of the attractor (when object = 2) is given more
% importance.
alpha = 0.85;
beta  = 0.15;
somAlphaBeta = [alpha, beta];

% Distribute the importance of alpha in the corresponding parameters
alphaVal = alpha/2;  
% Distribute the importance of beta in the corresponding parameters
betaVal = beta/6;  

m1 = 4;%10
m2 = 4;%11
sizeSOM = m1*m2;
                                                      
%% TRAINING VOCABULARY

if object == 1
    % Object 1: Favors follower's velocity
    [M, containerID, dataCode2, averageN,...
        covarianceN, containerNumbData, usedNeurons, containerData,...
        colorsMats] = ...
        somclustering(alphaVal, betaVal, data1, m1, m2);     
    
    %% Plot approximated data by SOM in xDot and ydot
    colSOM = colorsMats.SOM(usedNeurons,:);
    avNmat = cell2mat(averageN);
    h2 = figure;
    hold on;
    % Draws neuron with z=v_x
    s2 = scatter(avNmat(:,1), avNmat(:,2),50,'k', 'filled'); 
    % Data on the plane
    scatter(data1(:,1),data1(:,2),4,colorsMats.Data, 'filled');
    xlab = xlabel('$x$','interpreter','latex');
    ylab = ylabel('$y$','interpreter','latex');
    zlab = zlabel('$\dot{x}$','interpreter','latex');
    xlab.FontSize = 22;
    ylab.FontSize = 22;
    zlab.FontSize = 22;
    grid minor
    legend(s2,'Prototypes projected onto $\dot{x}$','interpreter','latex');
else
    %   Object 2: Favors attractor's velocity
    [M, containerID, dataCode2, averageN,...
        covarianceN, containerNumbData, usedNeurons, containerData,...
        colorsMats] = ...
        somclustering(alphaVal, betaVal, data2, m1, m2);

    %% Plot approximated data by SOM in xDot and ydot
    %   xDot components
    colSOM = colorsMats.SOM(usedNeurons,:);
    avNmat = cell2mat(averageN);
    h2 = figure;
    hold on;
    s2 = scatter(avNmat(:,1), avNmat(:,2),80,'k', 'filled');       %   Draws neuron with z=v_x
    scatter(data2(:,1),data2(:,2),...                         %   Data on the plane
        4,colorsMats.Data, 'filled');
    xlab = xlabel('$x$','interpreter','latex');
    ylab = ylabel('$y$','interpreter','latex');
    zlab = zlabel('$\dot{x}$','interpreter','latex');
    xlab.FontSize = 22;
    ylab.FontSize = 22;
    zlab.FontSize = 22;
    grid minor
    legend(s2,'Prototypes projected onto $\dot{x}$','interpreter','latex');

end

%% Create overall vocabulary
if object == 2
    % Load information from clusters related to SOM that favors follower's velocity
    load('vocabularyGen1.mat')                                                  
    dataCode1 = vocabularyGen.dataCode;
    % Occurrence of all possible combinations of coupled superstates (neurons)
    % transMat = Transition matrix_superstate
    % label    = codebook of events
    [label,transMat] = transitionProbability(dataCode1,dataCode2,...            
        sizeSOM,numberData);
    % Transition matrix of coupled superstates (neurons)
    transMatsTimeFinal = ...                  
        transitionTim(label,dataCode1,dataCode2, sizeSOM,transMat,numberData);
else
    transMat = [];
    transMatsTimeFinal = [];
end

%% Save outputs
%   Prototypes of states vectors
vocabularyGen.prototypes = M;             
%   Transition matrix of model
vocabularyGen.transitionMatrix = transMat;           
%   Cell with indexes of data in neurons
vocabularyGen.containerID = containerID;        
%   Array of crossed neurons
vocabularyGen.dataCode = dataCode2;    
%   Number of neurons in the SOM
vocabularyGen.sizeSom = sizeSOM;    
%   Average of neuron
vocabularyGen.averageN =  averageN;          
%   Covariance of neuron
vocabularyGen.covarianceN = covarianceN;  
%   For each couple of superstates is percentage of having it after a time instant
vocabularyGen.countDataN = containerNumbData;    
%   Labels of employed neurons
vocabularyGen.usedNeuronsLab = usedNeurons;   
%   Array of data inside each neuron
vocabularyGen.containerData = containerData;    
%   Matrix with histograms of time
vocabularyGen.transMatsTime = transMatsTimeFinal;       
%   Parameters used to weight the SOM
vocabularyGen.somAlphaBeta = somAlphaBeta;                                  

% Saving the codebook
if(object == 2)
    vocabularyGen.label = label;
end

filename = ['vocabularyGen' num2str(object) '.mat'];
save (filename,'vocabularyGen')
