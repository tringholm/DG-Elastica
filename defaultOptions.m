function options = defaultOptions(b,s)

% defaultOptions(b,s) provides an option struct to pass to the eulerElastica
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
% Struct fields:
%
% epsilon       - Smoothing parameter. Lower values = slower convergence 
%                 but possibly sharper images.
% stepSize      - Constant stepsize used in non-adaptive version. Also
%                 starting stepsize for adaptive version.
% maxIterations - Maximum number of iterations in the algorithm
% residualTol   - Stopping criterion tolerance in terms of relative 
%                 energy decrease (E(u(1)) - E(u(k))/E(u(1). 
%                 Lower values = more exact solution.
% scalarTol     - Error tolerance for scalar subproblems. If algorithm does
%                 not converge on lower values of residualTol, try lowering 
%                 this.
%
% makePlots     - Boolean for toggling plots on/off during runtime.
% saveOutput    - Boolean for toggling automatic output image saving on/off.
% outputName    - String containing name for automatic output.
% showDetails   - Boolean for toggling iteration details in terminal.
% 
% adaptivity    - Boolean for toggling adaptive stepsizes on/off.
% rho           - Decrease factor for adaptive stepsizes.
% gamma         - Increase factor for adaptive stepsizes.
% c             - Constant for accepting step sizes. 
%                 Lower value = more restrictive.
% 
% useMex        - Boolean for toggling MEX or MATLAB implementation.
% useParallel   - Boolean for toggling parallel implementation on/off. 
%                 Note: Parallel algorithm is MEX only.
% blockCount    - Number of blocks used for parallel splitting.
%
% Torbjørn Ringholm
% Email           : torbjorn.ringholm@ntnu.no
% Last updated    : 06/10/2017

%---------------------------------------------- parameters for convergence

options.epsilon = 1E-2;               % smoothing parameter
options.stepSize = 0.05/b;            % starting time step
options.maxIterations = 1000;         % max no. of iterations
options.residualTol = 1E-5;           % tolerance in rel. energy residual
options.scalarTol = 1E-5;             % tolerance for scalar subprob's

%---------------------------------------------- parameters for outputs

options.makePlots = 1;                % toggle plots while running
options.saveOutput = 0;               % toggle output image saving
options.outputName = 'output.png';    % name for output image
options.showDetails = 0;              % toggle iterations details

%---------------------------------------------- parameters for adaptivity

options.adaptivity = 1;               % toggle step size adaptivity 
if s == 1
    options.rho = 3;                  % decrease factor
    options.c = 0.1;                  % Wolfe condition constant
    options.gamma = 1.1;              % increase factor
else 
    options.rho = 4;                  % decrease factor
    options.c = 0.5;                  % Wolfe condition constant
    options.gamma = 1.1;              % increase factor
end

%---------------------------------------------- parameters for MEX/parallel

options.useMex = 1;                   % use MEX implementation (faster)
options.useParallel = 0;              % use parallel algorithm (MEX only)
options.blockCount = 20;              % for parallel, no. of blocks
end