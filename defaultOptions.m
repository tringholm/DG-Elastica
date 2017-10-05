function options = defaultOptions(b,s)

% defaultOptions provides an option struct to pass to the eulerElastica
% function with fields for tuning algorithm variables, choosing parallel
% and/or MEX implementation, and toggling outputs
%
% Input:
% b       - curvature term regularization weight
% s       - fidelity term is computed in L^s norm
%
% Output:
% options - struct with algorithm parameters
%
% Torbjørn Ringholm
% Email           : torbjorn.ringholm@ntnu.no
% Last updated    : 05/10/2017

%---------------------------------------------- parameters for convergence

options.epsilon = 1E-4;               % smoothing parameter
options.stepSize = 0.05/b;            % starting time step
options.maxIterations = 1000;         % max no. of iterations
options.residualTol = 1E-5;           % tolerance in rel. energy residual
options.scalarTol = 1E-5;             % tolerance for scalar subprob's

%---------------------------------------------- parameters for outputs

options.makePlots = 0;                % toggle plots while running
options.saveOutput = 0;               % toggle output image saving
options.outputName = 'output.png';    % name for output image
options.showDetails = 0;              % toggle iterations details

%---------------------------------------------- parameters for adaptivity

options.adaptivity = 1;               % toggle step size adaptivity 
if s == 1
    options.rho = 3;            
    options.c = 0.1;
    options.gamma = 1.1;
else
    options.rho = 4;
    options.c = 0.5;
    options.gamma = 1.1;
end

%---------------------------------------------- parameters for MEX/parallel

options.useMex = 0;                   % use MEX implementation (faster)
options.useParallel = 0;              % use parallel algorithm (MEX only)
options.blockCount = 40;              % for parallel, no. of blocks
end