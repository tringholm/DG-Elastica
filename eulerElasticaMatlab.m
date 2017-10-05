function [u, energy] = eulerElasticaMatlab(g,K,a,b,s,options)

% eulerElasticaMatlab computes an Euler's elastica regularized denoised 
% and/or inpainted image using a serial pure MATLAB implementation.  
%
% Input:
% g       - noisy input greyscale image, scaled from 0 to 1
% K       - logical map with false values on pixels to be inpainted
% a       - total variation regularization weight
% b       - curvature term regularization weight
% s       - fidelity term is computed in L^s norm
% options - struct with algorithm parameters, described in defaultOptions.m
%
% Output:
% u       - output image
% energy  - energy history
%
% Torbjørn Ringholm
% Email           : torbjorn.ringholm@ntnu.no
% Last updated    : 03/10/2017

%---------------------------------------------- extract arguments

epsilon = options.epsilon;
dt = options.stepSize;
maxit = options.maxIterations;
restol = options.residualTol;
xtol = options.scalarTol;
doplot = options.makePlots;
dooutput = options.saveOutput;
outputName = options.outputName;
dodetails = options.showDetails;
adapt = options.adaptivity;
c = options.c;
rho = options.rho;
gamma = options.gamma;
[Ny,Nx] = size(g);

%---------------------------------------------- show initial image
if doplot
    imfig = figure;
    nrgfig = figure;
    figure; imagesc(g); colormap(gray);
end

%---------------------------------------------- initialize variables
u = K.*g + (1-K).*rand(size(g));
energy = zeros(1,maxit+1);
energy(1) = energyFxn(u,K,g,a,b,s,epsilon);
u_old = zeros(size(u));
u_old(:) = u(:);
residual = 1;
t1 = tic;

%---------------------------------------------- timestep until convergence
for tstep = 1:maxit
    if residual < restol
        break
    end
    
    %------------------------------------------ compute gradient
    if adapt
        gg = gradTV(u,epsilon);
        gc = gradCurv(u,epsilon);
        if s == 2
            gradf = (K.*(u - g) + a*gg + b*gc);
        elseif s == 1
            gradf = (K.*((u > g) - (u < g)) + a*gg + b*gc);
        end
    end
    
    %------------------------------------------ one timestep
    u = dgstep(u,K,g,dt,a,b,s,epsilon,xtol);
    u = u(2:end-1,2:end-1);
    energy(tstep+1) = energyFxn(u,K,g,a,b,s,epsilon);
    residual = (energy(tstep) - energy(tstep+1))/energy(1);
    
    %------------------------------------------ check Wolfe cond
    if adapt
        if energy(tstep+1) > energy(tstep) + c*sum(sum(gradf.*(u - u_old)))
            dt = dt/rho;
        else
            dt = dt*gamma;
        end
    end
    
    %------------------------------------------ intermediate plots
    if doplot
        figure(imfig), imagesc(u); colormap(gray); pause(0.01)
        figure(nrgfig), plot(energy(1:tstep+1));  ylabel('Energy'); xlabel('Iteration'); pause(0.01)
    end
    if dodetails
        disp(['Iteration no. ' int2str(tstep) ':    Energy: ' num2str(energy(tstep+1),'%10.5e\n') '   Residual: ' num2str(residual,'%10.5e\n')]);
    end
    
    %------------------------------------------ prepare for next step
    u_old(:) = u(:);
end
t2 = toc(t1);

%---------------------------------------------- finalize
disp(['Algorithm DG stopped with residual: ' num2str(residual)])
disp(['Total time: ' num2str(t2) ' seconds. Iterations: ' int2str(tstep)])
disp(['Starting energy: ' num2str(energy(1)) '. Stopping energy: ' num2str(energy(tstep))])
if dooutput
    imwrite(u,outputName);
end
if doplot
    figure(imfig); title([' a: ' num2str(a) ' b: ' num2str(b)]);
end
end