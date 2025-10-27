
%% Path to trajectories

clc
clear
close all
addpath('./dataset')

%% Choosing training or testing, attractor or follower

training  = true; % do you want to select the training or testing data?
attractor = false; % do you want to select the attractor data or the follower data?

if training == true & attractor == false
    load('train_positions_follower.mat')
    datacell = train_positions_follower;
elseif training == true & attractor == true
    load('train_positions_attractor.mat')
    datacell = train_positions_attractor;
elseif training == false & attractor == false
    load('test_positions_follower.mat')
    datacell = test_positions_follower;
elseif training == false & attractor == true
    load('test_positions_attractor.mat')
    datacell = test_positions_attractor;
end

%% Find length of each trajectory and total length
num_trajectories = size(datacell,2);
totalLength = 0;
TrajectorySize = zeros(num_trajectories,1);
for i = 1:num_trajectories
    currentTraj = datacell{1,i};
    currentLength = size(currentTraj,1);
    TrajectorySize(i) = currentLength;
    totalLength = totalLength + currentLength;
end

% plot all trajectories
figure; % follower trajectories (training)
scatter(currentTraj(:,1),currentTraj(:,2),'b');
hold on
grid on
box on
title('First trajectory of set');
xlabel('x');
ylabel('y');
hold off


%% UMKF  (Null Force Filter)

% Transition matrix for the continous-time system.
A = [1  0  0  0;
    0  1  0  0;
    0  0  0  0;
    0  0  0  0];

% Process noise variance
sig_a = 0.2;
Q = eye(4) * sig_a^2;

% Measurement model.
H = [1 0 0 0;
    0 1 0 0];

% Variance in the measurements.
r1 = 1e-1;
R = diag([r1 r1]);
% Final state dimension is 4: position x, position y, velocity x, velocity y
state_dim = 4;
% Space for the estimates
MM = zeros(totalLength, state_dim); % mean matrix
Abnormalinn = zeros(totalLength, 1);
stateNoise = zeros(totalLength, state_dim);
index = 1; % index the rows of MM matrix (state matrix)
Traj = zeros(totalLength, state_dim); % contains observation on x and y and the computed innovation

for i = 1:num_trajectories
   % select one trajectory 
    Y = datacell{i}';                              % Y is the matrix with the observations of position x and y of followers
    % Filtering steps
 
    for j = 1:size(Y,2)
         % Initial guesses for the state mean and covariance.
        if j ==1   
            m = [Y(1,1) Y(2,1) 0 0]';
            P = diag([0.1 0.1 0.1 0.1]);
        end
              
        % Prediction
        [m,P] = kf_predict(m,P,A,Q);
        % Update
        [m,P,K,inn,IM,~,~,Abnormalinn1] = kf_update(m,P,Y(:,j),H,R);
        MM(index,1:2) = m(1:2,1)'; % position on x and y
        Traj(index,1:2) = Y(:,j)';
        MM(index,3:4) = inn'; % we can see the innovation as the velocity with the assumption of static object
        Traj(index,3:4) = inn';
        Abnormalinn(index,1) = Abnormalinn1(1,1)';
        stateNoise(index,:) = mvnrnd(m(:,1),P(:,:,1));    % add noise in velocities
        index = index + 1;
         
    end
   
end

%% randomly select a trajectory
a = 1;
b = num_trajectories;
rnd = randi([a b],1,1);   % for avoiding decimal number

% select data corresponding to each trajectory
state_rndCell = cell(1,num_trajectories);
StartPoint = 0;
AbnormalinnCell = cell(1,num_trajectories);
stateNoiseCell = cell(1,num_trajectories);
TrajCell = cell(1,num_trajectories);
for j = 1:1:num_trajectories
    EndPoint = TrajectorySize(j,1);
    e = EndPoint+StartPoint;
    state_rndCell{j} = MM(StartPoint+1:e,:);
    AbnormalinnCell{j} = Abnormalinn(StartPoint+1:EndPoint+StartPoint,:);
    stateNoiseCell{j} = stateNoise(StartPoint+1:EndPoint+StartPoint,:);
    TrajCell{j}= Traj(StartPoint+1:EndPoint+StartPoint,:);
    StartPoint = e;
end
state_rnd = state_rndCell{rnd};
Abnormalinn =  AbnormalinnCell{rnd};
stateNoise = stateNoiseCell{rnd};

%% Innovation
figure; % trajectory, filtered trajectory, velocity
subplot(1,2,1)
scatter(state_rnd(:,1),state_rnd(:,2))
title('Random selected trajectory')
xlabel('x')
ylabel('y')
subplot(1,2,2)
hold on
plot(state_rnd(:,3), 'r');
plot(state_rnd(:,4), 'b');
min_value = min(min(state_rnd(:, 3)), min(state_rnd(:, 4)));
max_value = max(max(state_rnd(:, 3)), max(state_rnd(:, 4)));
axis([0 size(state_rnd, 1), min_value, max_value])
title('Error');
legend('Error on x','Error on y');
xlabel('time instant');
grid on
box on
hold off

%% Save information
data.datacell = datacell;
data.MMCell = state_rndCell;
data.TrajCell = TrajCell;
data.stateNoiseCell = stateNoiseCell;
data.TrajectorySize = TrajectorySize;
data.MM = MM;
data.Trajectory = Traj;
data.TrajectoryNum = rnd;
data.CurrentTrajectory = state_rnd;
data.num_trajectories = num_trajectories;

%% SAVING
if training == true && attractor == true
    save('dataAttractor.mat','data')
elseif training == true && attractor == false
    save('dataFollower.mat','data')
elseif training == false && attractor == true
    save('dataAttractorTest.mat','data')
else
    save('dataFollowerTest.mat','data')
end