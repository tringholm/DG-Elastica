function [u, energy] = Elastica_MEX_para_selfcontained()
%%% FUNCTION FOR TESTING, NEEDS IMAGES TO RUN %%%

mex partitionsteprow_mex2.c
mex dgstepblock_mex2.c

%%%%%%%%% LOUVRE SALT & PEPPER DENOISING %%%%%%%%%
% Load image, convert to grayscale and apply salt and pepper noise
rng(9)
image = 'Louvrebig.png';
gl = imread(image);
gl = double(rgb2gray(gl));
gl = gl./max(max(gl));
g = imnoise(gl,'salt & pepper',0.25);
% Domain filter is full domain. Initialize step sizes
K = ones(size(g));
[Ny,Nx] = size(g);
dx = 1/Nx;
dy = 1/Ny;

% Set regularization parameters and time step size
%%% ELASTICA %%%
alpha = 1;
a = 6.5E-4;
b = 1E-10;
dt = dx*dy*100000;
%%% TV %%%
% alpha = 1;
% a = 0.001;
% b = 0;
% dt = dx*dy*100000;
%%%%%%%%%%%%%%%%
% Initial guess for image and use L_1 norm for fidelity term
u = g;
s = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%% FLOWER INPAINTING %%%%%%%%%%%%%
%         
% % Load image, convert to grayscale and destroy 95% of pixels randomly
% image = 'flowerfull.png';
% gl = imread(image);
% g = gl;
% rng(12);
% redfact = 0.95;
% for i = 1:size(g,1)
%     for j = 1:size(g,2)
%         r = rand();
%         if r > 1-redfact
%             g(i,j,:) = 255;
%         end
%     end
% end
% 
% % Set domain filter to known pixels. Initialize step sizes
% K = 1-(g(:,:,1) == 255 & g(:,:,2) == 255 & g(:,:,3) == 255);
% g = double(rgb2gray(g));
% g = double(g)./max(max(double(g)));
% [Ny,Nx] = size(g);
% dx = 1/Nx;
% dy = 1/Ny;
% 
% % Set regularization parameters and time step size
% %%%% ELASTICA %%%%
% alpha = 1;
% a = 1E-9;
% b = 1E-15;
% dt = dx*dy*100000000000;
% 
% %%%% TV %%%%%
% % alpha = 1;
% % a = 5E-10;
% % b = 0;
% % dt = dx*dy*100000000000;
% %%%%%%%%%%%%%%
% 
% % Initial guess for image is random in inpainting domain. Use L_2 norm.
% u = K.*g + (1-K).*rand(size(g));
% s = 2;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


T = 1000; % Max no. of time steps
epsilon = 1E-16;
restol = 1E-6; % Stopping criterion tolerance
xtol = 1E-5;
Nblocks = 40;

figure(1); imagesc(g); colormap(gray);


energy = zeros(1,T+1);
energy(1) = energyfxn(u,K,g,dx,dy,a,b,alpha,s,epsilon);


rPerBlock = ceil(Ny/Nblocks);
rows = rPerBlock:rPerBlock:Ny-1;
upart = zeros(4,Nx,length(rows));
upart2 = zeros(8,Nx,length(rows));
Kpart = zeros(4,Nx,length(rows));
gpart = zeros(4,Nx,length(rows));

residual = 1;
t1 = tic;
imfig = figure;
% nrgfig = figure;
for tstep = 1:T
    if residual < restol
        residual
        break
    end
    tic
    
    for i = 1:length(rows)
        row = rows(i);
        upart2(:,:,i) = u(row-3:row+4,:);
        Kpart(:,:,i) = K(row-1:row+2,:);
        gpart(:,:,i) = g(row-1:row+2,:);
    end
    
    parfor i = 1:length(rows)
        upart(:,:,i) = partitionsteprow_mex2(upart2(:,:,i),Nx,4,gpart(:,:,i),dx,dy,dt,a,b,alpha,s,epsilon,xtol,Kpart(:,:,i));
    end
    
    for i = 1:length(rows)
        row = rows(i);
        u(row-1:row+2,:) = upart(:,:,i);
    end
    
    disp(['Time used in partition: ' num2str(toc)]);
    
    dgstepinl = @(z) dgstepblock_mex2(z,K,g,dx,dy,dt,a,b,alpha,s,epsilon,xtol);
    tic
    u = blockproc(u,[rPerBlock,Nx],dgstepinl,'BorderSize',[2,1],'UseParallel',true);
    disp(['Time used in sweep: ' num2str(toc)])
    
%     dt = dt*tstep/(tstep+1);
%     u_collection(:,:,tstep+1) = u;
    energy(tstep+1) = energyfxn(u,K,g,dx,dy,a,b,alpha,s,epsilon);
    residual = (energy(tstep) - energy(tstep+1))/energy(1);
%     if residual < 0
%         keyboard
%     end
    clf(imfig);
    figure(imfig), imagesc(u); colormap(gray);
%     truesize(imfig);
%     truesize(5,[2*Ny,2*Nx]);
    pause(0.01)
%     clf(nrgfig);
    figure(3), plot(energy(1:tstep+1));
    pause(0.01)
end
t2 = toc(t1);
if exist('gl','var')
    figure(4);
    imagesc(gl);
%     truesize(4);
    colormap(gray);
end
disp(['Time used: ' num2str(t2)])
beep
imwrite(u,'output.png');
figure(imfig); title(['alpha: ' num2str(alpha) ' a: ' num2str(a) ' b: ' num2str(b)]);
end

function E = energyfxn(u,K,g,dx,dy,a,b,alpha,s,epsilon)
[Ny,Nx] = size(u);
Dxu = zeros(Ny,Nx+1);  % x-derivatives at i+1/2,j points
Dyu = zeros(Ny+1,Nx);  % y-derivatives at i,j+1/2 points
Dxu2 = zeros(Ny+1,Nx); % x-derivatives at i,j+1/2 points
Dyu2 = zeros(Ny,Nx+1); % y-derivatives at i+1/2,j points

Dxu(:,2:end-1) = diff(u,1,2)/dx;
Dxu2(2:end-1,:) = 1/4*(Dxu(1:end-1,1:end-1) + Dxu(2:end,1:end-1) + Dxu(1:end-1,2:end) + Dxu(2:end,2:end));
Dxu2(1,:) = 1/2*(Dxu(1,1:end-1) + Dxu(1,2:end));
Dxu2(end,:) = 1/2*(Dxu(end,1:end-1) + Dxu(end,2:end));

Dyu(2:end-1,:) = diff(u)/dy;
Dyu2(:,2:end-1) = 1/4*(Dyu(1:end-1,1:end-1) + Dyu(2:end,1:end-1) + Dyu(1:end-1,2:end) + Dyu(2:end,2:end));
Dyu2(:,1) = 1/2*(Dyu(1:end-1,1) + Dyu(2:end,1));
Dyu2(:,end) = 1/2*(Dyu(1:end-1,end) + Dyu(2:end,end));

gradabs = realsqrt(Dxu.^2 + Dyu2.^2 + epsilon);
gradabs2 = realsqrt(Dxu2.^2 + Dyu.^2 + epsilon);

Dx = Dxu./gradabs;
Dy = Dyu./gradabs2;

D1 = diff(Dx,1,2)/dx;
D2 = diff(Dy)/dy;

gradabsmid = realsqrt(Dxu(:,1:end-1).^2 + Dyu(1:end-1,:).^2 + epsilon);

E = alpha*dx*dy*sum(sum((a + b*(D1 + D2).^2).*gradabsmid)) + dx*dy/s*sum(sum((abs(K.*(u-g))).^s));
end