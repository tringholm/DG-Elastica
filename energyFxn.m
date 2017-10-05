function E = energyFxn(u,K,g,a,b,s,epsilon)

% energyFxn computes the Euler's elastica energy discretized on a staggered
% grid
%
% Input:
% u       - current restored image
% K       - logical map with false values on pixels to be inpainted
% g       - noisy input greyscale image
% a       - total variation regularization weight
% b       - curvature term regularization weight
% s       - fidelity term is computed in L^s norm
% epsilon - smoothing parameter
%
% Output:
% E       - discretized Euler's elastica energy
%
% Torbjørn Ringholm
% Email           : torbjorn.ringholm@ntnu.no
% Last updated    : 03/10/2017

%---------------------------------------------- derivative approximations
[Ny,Nx] = size(u);
Dxu = zeros(Ny,Nx+1); 
Dyu = zeros(Ny+1,Nx);  
Dxu2 = zeros(Ny+1,Nx); 
Dyu2 = zeros(Ny,Nx+1); 
Dxu(:,2:end-1) = diff(u,1,2);
Dyu(2:end-1,:) = diff(u);

%---------------------------------------------- derivative interpolations
Dxu2(2:end-1,:) = 1/4*(Dxu(1:end-1,1:end-1) + Dxu(2:end,1:end-1) + Dxu(1:end-1,2:end) + Dxu(2:end,2:end));
Dxu2(1,:) = 1/2*(Dxu(1,1:end-1) + Dxu(1,2:end));
Dxu2(end,:) = 1/2*(Dxu(end,1:end-1) + Dxu(end,2:end));
Dyu2(:,2:end-1) = 1/4*(Dyu(1:end-1,1:end-1) + Dyu(2:end,1:end-1) + Dyu(1:end-1,2:end) + Dyu(2:end,2:end));
Dyu2(:,1) = 1/2*(Dyu(1:end-1,1) + Dyu(2:end,1));
Dyu2(:,end) = 1/2*(Dyu(1:end-1,end) + Dyu(2:end,end));

%---------------------------------------------- gradient moduli
gx = realsqrt(Dxu.^2 + Dyu2.^2 + epsilon);
gy = realsqrt(Dxu2.^2 + Dyu.^2 + epsilon);
gm = realsqrt(Dxu(:,1:end-1).^2 + Dyu(1:end-1,:).^2 + epsilon);

%---------------------------------------------- curvature term
Dx = Dxu./gx;
Dy = Dyu./gy;
kx = diff(Dx,1,2);
ky = diff(Dy);

%---------------------------------------------- final energy
E = sum(sum((a + b*(kx + ky).^2).*gm)) + 1/s*sum(sum((K.*abs(u-g)).^s));
end