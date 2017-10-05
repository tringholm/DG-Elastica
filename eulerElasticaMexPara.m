function [u, energy] = eulerElasticaMexPara(g,K,a,b,s,options)

% eulerElasticaMexPara computes an Euler's elastica regularized denoised 
% and/or inpainted image using a parallel MEX function for the time steps.  
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
Nblocks = options.blockCount;
[Ny,Nx] = size(g);

%---------------------------------------------- show initial image
if doplot
    imfig = figure;
    nrgfig = figure;
    figure; imagesc(g); colormap(gray);
end

%---------------------------------------------- initialize parallel vars
rPerBlock = ceil(Ny/Nblocks);
rows = rPerBlock:rPerBlock:Ny-1;
rows2 = [2 rows Ny-2];
upart = zeros(4,Nx,length(rows));
upart2 = zeros(8,Nx,length(rows));
Kpart = zeros(4,Nx,length(rows));
gpart = zeros(4,Nx,length(rows));
if adapt
    gc = zeros(size(g));
    gcloc = zeros(rPerBlock+4,Nx,length(rows2)-3);
end

%---------------------------------------------- initialize variables
u = K.*g + (1-K).*rand(size(g));
energy = zeros(1,maxit+1);
energy(1) = energyFxn(u,K,g,a,b,s,epsilon);
u_old = zeros(size(u));
u_old(:) = u;
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
        %-------------------------------------- first block
        indl = 1;
        indh = rows2(2)+2;
        gcloc1 = gradCurvMexPara(u(indl:indh,:),epsilon,Nx,indh - indl +1,Ny,indl-1);
        gc(1:rows2(2),:) = gcloc1(1:end-2,:);
        
        %-------------------------------------- other blocks
        parfor i = 2:length(rows2)-2
            indl = (i-1)*rPerBlock -1;
            indh = i*rPerBlock+2;
            gcloc(:,:,i-1) = gradCurvMexPara(u(indl:indh,:),epsilon,Nx,indh - indl +1,Ny,indl-1);
        end
        for i = 2:length(rows2)-2
            gc(rows2(i)+1:rows2(i+1),:) = gcloc(3:end-2,:,i-1);
        end
        
        %-------------------------------------- last block
        indl = rows2(end-1)-1;
        indh = rows2(end)+2;
        gcloc2 = gradCurvMexPara(u(indl:indh,:),epsilon,Nx,indh - indl +1,Ny,indl-1);
        gc(rows2(end-1)+1:end,:) = gcloc2(3:end,:);
        
        if s == 2
            gradf = K.*(u - g) + a*gg + b*gc;
        elseif s == 1
            gradf = K.*((u > g) - (u < g)) + a*gg + b*gc;
        end
    end
    
    %------------------------------------------ update partition info
    for i = 1:length(rows)
        row = rows(i);
        upart2(:,:,i) = u(row-3:row+4,:);
        Kpart(:,:,i) = K(row-1:row+2,:);
        gpart(:,:,i) = g(row-1:row+2,:);
    end
    
    %------------------------------------------ process borders
    parfor i = 1:length(rows)
        upart(:,:,i) = partitionMex(upart2(:,:,i),Nx,4,gpart(:,:,i),dt,a,b,s,epsilon,xtol,Kpart(:,:,i));
    end
    for i = 1:length(rows)
        row = rows(i);
        u(row-1:row+2,:) = upart(:,:,i);
    end
    
    %------------------------------------------ process blocks
    dgstepinl = @(z) dgstepMexPara(z,K,g,dt,a,b,s,epsilon,xtol);
    u = blockproc(u,[rPerBlock,Nx],dgstepinl,'BorderSize',[2,1],'UseParallel',true);    
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
disp(['Algorithm DG-PARA stopped with residual: ' num2str(residual)])
disp(['Total time: ' num2str(t2) ' seconds. Iterations: ' int2str(tstep)])
disp(['Starting energy: ' num2str(energy(1)) '. Stopping energy: ' num2str(energy(tstep))])
if dooutput
    imwrite(u,outputName);
end
if doplot
    figure(imfig); title([' a: ' num2str(a) ' b: ' num2str(b)]);
end
end