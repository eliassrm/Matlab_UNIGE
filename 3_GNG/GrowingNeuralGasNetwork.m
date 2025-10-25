
%% GNG function
function net = GrowingNeuralGasNetwork(inputData,params,ctitle)

if ~exist('PlotFlag', 'var')
    PlotFlag = true;
end

%% Standardize data
meanData = mean(inputData);
stdData  = std(inputData);
dataNorm = inputData - meanData;
CV       = dataNorm./repmat(stdData,size(inputData,1),1);

%% Weighting data
CV = [CV(:,1:2)*params.betaWeight CV(:,3:4)*params.alphaWeight];
inputNormOrd = CV;

%% Shuffle Data
nData = size(CV,1);                                                         % Size of input data (number of training samples)
nDim  = size(CV,2);                                                          % Dimension of input data
seed  = RandStream('mt19937ar','Seed',0);

CV    = CV(randperm(seed,nData), :);                                                % Random permutation of input data

CVmin = min(CV);
CVmax = max(CV);

%% Parameters
N = params.N;                                                               % number of nodes in GNG
MaxIt = params.MaxIt;
L = params.L;
epsilon_b = params.epsilon_b;                                               % stepsize to update weight of winner node
epsilon_n = params.epsilon_n;                                               % stepsize to update weight of all direct neibours
alpha = params.alpha;
delta = params.delta;
T = params.T;

%% Initialization
% Initial 2 nodes for training the algorithm  
Ni = 2; 
% Vectors of centers of clusters
wNorm = zeros(Ni, nDim);
for i = 1:Ni
    wNorm(i,:) = unifrnd(CVmin, CVmax);                                         % It returns an array of random numbers generated from the continuous uniform distributions with lower and upper endpoints specified by 'CVmin' and 'CVmax'.
end
% Errors
E = zeros(Ni,1);
% Connections
C = zeros(Ni, Ni);
% Ages
t = zeros(Ni, Ni);

%% Loop

nx = 0;

for it = 1:MaxIt                                                            %   Number of iteration
    for c = 1:nData                                                         %   Number of data
        % Select Input
        nx = nx + 1;                                                        %   Counter of cycles inside the algorithm
        x = CV(c,:);                                                        %   pick first input vector from permuted inputs
        
        % Competion and Ranking
        d = pdist2(x, wNorm,'euclidean');                                   %   pdist- Pairwise distance between two sets of observations(Eucledian distance between input and 2 nodes initialised before)
        [~, SortOrder] = sort(d);                                           %   Organize distances between nodes and the first data point in an ascending order
        s1 = SortOrder(1);                                                  %   Closest node index to the first data point   
        s2 = SortOrder(2);                                                  %   Second closest node index to the first data point
        
        % Aging
        t(s1, :) = t(s1, :) + 1;                                            %   Increment the age of all edges emanating from s1
        t(:, s1) = t(:, s1) + 1;
        
        % Add Error
        E(s1) = E(s1) + d(s1)^2;                                            %   Squared distance between input signal and nearest unit in input space to a local counter variable
        
        % Adaptation
        wNorm(s1,:) = wNorm(s1,:) + epsilon_b*(x-wNorm(s1,:));                          %   Move the nearest distance node and it's neibors to wards input signal by fractions Eb and En resp.
        Ns1 = find(C(s1,:) == 1);                                           %   Take all the connections of the closest node to the data in question   
        for j = Ns1
            wNorm(j,:) = wNorm(j,:) + epsilon_n*(x-wNorm(j,:));                         %   Move the direct topological neibors of nearest distance node (S1) and it's neibors to wards input signal by fractions Eb and En resp.
        end
        
        % Create Link
        C(s1,s2) = 1;                                                       %  If s1 and s2 are connected by an edge , set the age of this edge to zero , it such edge doesn't exist create it
        C(s2,s1) = 1;
        t(s1,s2) = 0;                                                       %   Age of the edge
        t(s2,s1) = 0;
        
        % Remove Old Links
        C(t > T) = 0;                                                       %   remove edges with an age larger than Amax(a threshold value)
        nNeighbor = sum(C);                                                 %   Number of conecctions of each node
        AloneNodes = (nNeighbor==0);                                        
        C(AloneNodes, :) = [];
        C(:, AloneNodes) = [];
        t(AloneNodes, :) = [];
        t(:, AloneNodes) = [];
        wNorm(AloneNodes, :) = [];
        E(AloneNodes) = [];
        
        % Add New Nodes
        if mod(nx, L) == 0 && size(wNorm,1) < N
            [~, q] = max(E);                                                %   Determine the unit q with the maximum accumulated error
            [~, f] = max(C(:,q).*E);                                        %   Maximum index related to the error related to a connected node
            
            r = size(wNorm,1) + 1;
            wNorm(r,:) = (wNorm(q,:) + wNorm(f,:))/2;                                   %   Insert a new unit r halfway between q and it's neibor f with the largest error variable
            
            %   Remove old connections and introduce the presence of the
            %   new created node
            C(q,f) = 0;
            C(f,q) = 0;                                                     
            C(q,r) = 1;
            C(r,q) = 1;
            C(r,f) = 1;
            C(f,r) = 1;
            t(r,:) = 0;
            t(:, r) = 0;
            
            E(q) = alpha*E(q);                                              %   Decrease the error variable of q and f by multiplying them with a constand 'alpha'
            E(f) = alpha*E(f);
            
            E(r) = E(q);
        end
        
        % Decrease Errors
        E = delta*E;                                                        %   Decrease error variables by multiplying them with a constant delta
    end
    
    % Plot Results
    if PlotFlag
        figure(1)
        PlotResults(CV, wNorm, C);
        pause(0.01);
    end
end

%% Export Results data samples in nodes
%% Denormalization of GNGN generated centroids
wNorm = [wNorm(:,1:2)/params.betaWeight wNorm(:,3:4)/params.alphaWeight];
w = wNorm.*repmat(stdData,size(wNorm,1),1);
w = w + repmat(meanData,size(wNorm,1),1);

datanodes = cell(1,size(wNorm,1));
dataColorNode = [];
for c = 1:nData                                                            %   Number of data
    x = inputNormOrd(c,:);
    d = pdist2(x, wNorm,'euclidean') ;                                         %   pdist- Pairwise distance between two sets of observations(Eucledian distance between input and 2 nodes initialised before)
    [~, minNode] = min(d) ;                                               %   Organize distances between nodes and the first data point in an ascending order
    dataColorNode = [dataColorNode; minNode];

    x = inputData(c,:) ;     %not normalize and orderd data
    datanodes{1,minNode} = [datanodes{1,minNode}; x];
    
end

%% output  of GNG

%   Data inside each node
net.datanodes  = datanodes ;
net.N = size(datanodes,2);  %   number of nodes
net.wNorm = wNorm;   % protocol values
net.w = w;   % protocol values (calculated centroids)
net.E = E;   % error value
net.C = C;   % connections
net.t = t;   % connection values
net.Parameters = params;
net.Discrete_data = dataColorNode;
net.dataNorm = inputNormOrd;
net.data = inputData;

end


