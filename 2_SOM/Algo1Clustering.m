clc
clear
close all

addpath('./somtoolbox')

follower = false;

%% Load data set 

if follower == true
    % load training data here
    load('dataFollower.mat')
else
    % load training data here
    load('dataAttractor.mat')
end

InputData = data.MM;

%% TRAINING SOM
% Weights definition for position (beta), velocity (alpha) 
beta = 2;  
alpha = 0.5;                                            

% Selcet number of clusters by using m1 amd m2
m1 = 5;
m2 = 5;
SomSize = m1*m2;

% call SOM function for clustering
[w, Discrete_dataNode, Discrete_data, datanodes,...
    colorsMats,N] = somclustering(alpha, beta, InputData,m1,m2,SomSize);
usedNeurons = 1:N;

%% store information
net.data = InputData ;
net.w = w;
net.Discrete_dataNode = Discrete_dataNode;
net.Discrete_data = Discrete_data;
net.datanodes = datanodes;
net.N = N;
net.UMKF = data;

if follower == true
    save('VocabularySOMF.mat','net')
else
    save('VocabularySOMA.mat','net')
end

% Initialize best scores
best_score = inf;
best_alpha = 1;
best_beta = 1;


% % Parameter Grid
% alpha_values = [0.1, 0.5, 1, 2];
% beta_values = [0.1, 0.5, 1, 2];
% 
% % Loop over alpha and beta combinations
% for alpha = alpha_values
%     for beta = beta_values
%         % SOM Clustering
%         [w, Discrete_dataNode, Discrete_data, datanodes, colorsMats, N] = somclustering(alpha, beta, InputData, m1, m2, SomSize);
% 
%         % Calculate Quantization Error (QE)
%         QE = mean(min(pdist2(InputData, w), [], 2));  % Distance between data and closest neuron
% 
% 
%         % Update best parameters
%         if QE < best_score
%             best_score = QE;
%             best_alpha = alpha;
%             best_beta = beta;
%         end
% 
%         fprintf('Alpha: %.2f, Beta: %.2f, Score: %.4f\n', alpha, beta, QE);
%     end
% end
% 
% % Display Best Parameters
% fprintf('Best Alpha: %.2f, Best Beta: %.2f with Score: %.4f\n', best_alpha, best_beta, best_score);




%% Plotting
colSOM = colorsMats.SOM(usedNeurons,:);
avNmat = w;
h2 = figure;
hold on;
s2 = scatter3(avNmat(:,1), avNmat(:,2),avNmat(:,3),50,'k', 'filled');             %   Draws neuron with z=v_x
scatter3(InputData(:,1),InputData(:,2),InputData(:,3),...                         %   Data on the plane
    4,colorsMats.Data, 'filled');
xlab = xlabel('$x$','interpreter','latex');
ylab = ylabel('$y$','interpreter','latex');
zlab = zlabel('$\dot{x}$','interpreter','latex');
xlab.FontSize = 22;
ylab.FontSize = 22;
zlab.FontSize = 22;
grid minor
leg2 = legend(s2,'Prototypes projected onto $\dot{x}$');
set(leg2,'Interpreter','latex');
leg2.FontSize = 14;
leg2.Position = [0.3010    0.8293     0.4947    0.0543];
ax1 = gca;
ax1.View = [-38.4000 28.4000];
%h2.Position = [607   495   569   486];

%   Adjust image axes (Start)
outerpos = ax1.OuterPosition;
ti = ax1.TightInset;
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax1.Position = [left bottom ax_width-0.01 ax_height];
%   Adjust image axes (End)
set(h2,'Units','Inches');
posPage = get(h2,'Position');
set(h2,'PaperPositionMode','Auto','PaperUnits',...
    'Inches','PaperSize',[posPage(3), posPage(4)])
%   yDot components
h3 = figure;
hold on;
s3 = scatter3(avNmat(:,1), avNmat(:,2),avNmat(:,4),70,'k', 'filled');              %   Draws neuron with z=v_x
scatter3(InputData(:,1),InputData(:,2),InputData(:,4), 4,...
    colorsMats.Data, 'filled');
xlab = xlabel('$x$','interpreter','latex');
ylab = ylabel('$y$','interpreter','latex');
zlab = zlabel('$\dot{y}$','interpreter','latex');
xlab.FontSize = 22;
ylab.FontSize = 22;
zlab.FontSize = 22;
grid minor
leg3 = legend(s3,'Prototypes projected onto $\dot{y}$');
set(leg3,'Interpreter','latex');
leg3.FontSize = 14;
leg3.Position = [0.2912    0.75    0.4932    0.0543];
ax2 = gca;
ax2.View = [-54.8000   28.4000];
%h3.Position = [1215 495 569  486];

%   Adjust image axes (Start)
outerpos = ax2.OuterPosition;
ti = ax2.TightInset;
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
%ax2.Position = [left bottom ax_width-0.01 ax_height];

%   Adjust image axes (End)
set(h3,'Units','Inches');
posPage = get(h3,'Position');
set(h3,'PaperPositionMode','Auto','PaperUnits',...
    'Inches','PaperSize',[posPage(3), posPage(4)])