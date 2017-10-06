function [u, energy] = eulerElastica(g,a,b,s,options,K)

% [u, energy] = eulerElastica(g,a,b,s,options,K) computes an Euler's 
% elastica regularized denoised and/or inpainted image u using a 
% serial/parallel MATLAB/MEX implementation.  
%
% eulerElastica(g,a,b,s) computes a denoised image using default options.
% eulerElastica(g,a,b,s,options) computes a denoised image using tailored
% options.
% eulerElastica(g,a,b,s,options,K) computes an inpainted image usinge
% tailored options and an inpainting mask K. K has the dimension of g, with
% zeroes in pixels which are to be inpainted and ones elsewhere.
%
% Input:
% g       - noisy input greyscale image, scaled from 0 to 1
% a       - total variation regularization weight
% b       - curvature term regularization weight
% s       - fidelity term is computed in L^s norm
% options - struct with algorithm parameters, described in defaultOptions.m
% K       - logical map with false values on pixels to be inpainted
%
% Output:
% u       - output image
% energy  - energy history
%
% Torbjørn Ringholm
% Email           : torbjorn.ringholm@ntnu.no
% Last updated    : 06/10/2017

%---------------------------------------------- extract arguments
if nargin == 4
    disp('No options specified, using default options.')
    options = defaultOptions(b,s);
    K = ones(size(g));
elseif nargin == 5
    K = ones(size(g));
elseif nargin ~= 6
    disp('Too few or too many input arguments.')
    u = g;
    energy = inf;
    return
end

if options.useParallel
    if exist('dgstepMexPara','file') ~= 3
        mex dgstepMexPara.c
    end
    if exist('gradCurvMexPara','file') ~= 3
        mex gradCurvMexPara.c
    end
    if exist('partitionMex','file') ~= 3
        mex partitionMex.c
    end
    [u, energy] = eulerElasticaMexPara(g,K,a,b,s,options);
else
    if options.useMex
        if exist('dgstepMex','file') ~= 3
            mex dgstepMex.c
        end
        if exist('gradCurvMex','file') ~= 3
            mex gradCurvMex.c
        end
        [u,energy] = eulerElasticaMex(g,K,a,b,s,options);
    else
        [u,energy] = eulerElasticaMatlab(g,K,a,b,s,options);
    end
end

end