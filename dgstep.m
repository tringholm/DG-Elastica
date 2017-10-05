function u = dgstep(u,K,g,dt,a,b,s,epsilon,xtol)

% dgstep propagates one timestep of the discrete gradient algorithm
%
% Input:
% u       - current restored image
% K       - logical map with false values on pixels to be inpainted
% g       - noisy input greyscale image
% dt      - time step size
% a       - total variation regularization weight
% b       - curvature term regularization weight
% s       - fidelity term is computed in L^s norm
% epsilon - smoothing parameter
% xtol    - tolerance for Brent-Dekker nonlinear equation solver
%
% Output:
% u       - updated image after one time step
%
% Torbjørn Ringholm
% Email           : torbjorn.ringholm@ntnu.no
% Last updated    : 03/10/2017

%---------------------------------------------- derivative approximations
[Ny,Nx] = size(u);
Dxu = zeros(Ny,Nx+1);  % x-derivatives at i+1/2,j points
Dyu = zeros(Ny+1,Nx);  % y-derivatives at i,j+1/2 points
Dxu2 = zeros(Ny+1,Nx); % x-derivatives at i,j+1/2 points
Dyu2 = zeros(Ny,Nx+1); % y-derivatives at i+1/2,j points
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

%---------------------------------------------- padding for main loop
u = padarray(u,[1 1]);
Dxu = padarray(Dxu,[1 1]);
Dyu = padarray(Dyu,[1 1]);
Dxu2 = padarray(Dxu2,[1 1]);
Dyu2 = padarray(Dyu2,[1 1]);
gx = padarray(gx,[1 1],sqrt(epsilon));
gy = padarray(gy,[1 1],sqrt(epsilon));
gm = padarray(gm,[1,1],sqrt(epsilon));
Dx = padarray(Dx,[1 1]);
Dy = padarray(Dy,[1 1]);
kx = padarray(kx,[1 1]);
ky = padarray(ky,[1 1]);
tic

%---------------------------------------------- main loop
for i = 1:Ny
    for j = 1:Nx  
        %-------------------------------------- extract local values
        u_l = u(i:i+2,j:j+2);
        u_old = u_l(2,2);
        Dxu_l = Dxu(i:i+2,j:j+3);
        Dxul = Dxu_l(:,1:3);
        Dxuc = Dxu_l(:,2:3);
        Dxur = Dxu_l(:,2:4);
        Dxum = 0.25*(Dxul + Dxur);
        Dxumt = Dxum(1:2,:);
        Dxuml = Dxum(2:3,:);
        Dyu_l = Dyu(i:i+3,j:j+2);
        Dyut = Dyu_l(1:3,:);
        Dyuc = Dyu_l(2:3,:);
        Dyub = Dyu_l(2:4,:);
        Dyum = 0.25*(Dyut + Dyub);
        Dyuml = Dyum(:,1:2);
        Dyumr = Dyum(:,2:3);
        Dx_l = Dx(i:i+2,j:j+3);
        Dy_l = Dy(i:i+3,j:j+2);
        kx_l = kx(i:i+2,j:j+2);
        ky_l = ky(i:i+2,j:j+2);
        gm_l = gm(i:i+2,j:j+2);
        
        %-------------------------------------- old energy in pixel
        E = sum(sum((a + b*(kx_l + ky_l).^2).*gm_l));
        
        %-------------------------------------- update pixel value
        [u(i+1,j+1),~,~] = fzeroFast( ...
            u(i+1,j+1)+0.1,xtol,u_old,u_l,Dxul,Dxuc,Dxur,Dxumt,Dxuml,Dyut,Dyuc,Dyub,Dyuml,Dyumr,...
            gm_l,Dx_l,Dy_l,E,K(i,j),g(i,j),i,j,Ny,Nx,dt,a,b,s,epsilon);
        u_l(2,2) = u(i+1,j+1);
        
        %-------------------------------------- update deriv approximations
        if j > 1
            Dxu(i+1,j+1) = (u_l(2,2) - u_l(2,1));
        end
        if j < Nx
            Dxu(i+1,j+2) = (u_l(2,3) - u_l(2,2));
        end
        if i > 1
            Dyu(i+1,j+1) = (u_l(2,2) - u_l(1,2));
        end
        if i < Ny
            Dyu(i+2,j+1) = (u_l(3,2) - u_l(2,2));
        end
        
        %-------------------------------------- update deriv interpolations
        if i == 1
            Dxu2(i+1,j:j+2) = 1/2*(Dxu(i+1,j:j+2) + Dxu(i+1,j+1:j+3));
            Dxu2(i+2,j:j+2) = 1/4*(Dxu(i+1,j:j+2) + Dxu(i+2,j:j+2) + Dxu(i+1,j+1:j+3) + Dxu(i+2,j+1:j+3));
        elseif i == Ny
            Dxu2(i+1,j:j+2) = 1/4*(Dxu(i,j:j+2) + Dxu(i+1,j:j+2) + Dxu(i,j+1:j+3) + Dxu(i+1,j+1:j+3));
            Dxu2(i+2,j:j+2) = 1/2*(Dxu(i+1,j:j+2) + Dxu(i+1,j+1:j+3));
        else
            Dxu2(i+1:i+2,j:j+2) = 1/4*(Dxu(i:i+1,j:j+2) + Dxu(i+1:i+2,j:j+2) + Dxu(i:i+1,j+1:j+3) + Dxu(i+1:i+2,j+1:j+3));
        end
        
        if j == 1
            Dyu2(i:i+2,j+1) = 1/2*(Dyu(i:i+2,j+1) + Dyu(i+1:i+3,j+1));
            Dyu2(i:i+2,j+2) = 1/4*(Dyu(i:i+2,j+1) + Dyu(i+1:i+3,j+1) + Dyu(i:i+2,j+2) + Dyu(i+1:i+3,j+2));
        elseif j == Nx
            Dyu2(i:i+2,j+1) = 1/4*(Dyu(i:i+2,j) + Dyu(i+1:i+3,j) + Dyu(i:i+2,j+1) + Dyu(i+1:i+3,j+1));
            Dyu2(i:i+2,j+2) = 1/2*(Dyu(i:i+2,j+1) + Dyu(i+1:i+3,j+1));
        else
            Dyu2(i:i+2,j+1:j+2) = 1/4*(Dyu(i:i+2,j:j+1) + Dyu(i+1:i+3,j:j+1) + Dyu(i:i+2,j+1:j+2) + Dyu(i+1:i+3,j+1:j+2));
        end
        
        %-------------------------------------- update gradient moduli
        gx(i:i+2,j+1:j+2) = realsqrt(Dxu(i:i+2,j+1:j+2).^2 + Dyu2(i:i+2,j+1:j+2).^2 + epsilon);
        gy(i+1:i+2,j:j+2) = realsqrt(Dxu2(i+1:i+2,j:j+2).^2 + Dyu(i+1:i+2,j:j+2).^2 + epsilon);
        gm(i+1:i+2,j+1:j+2) = realsqrt(Dxu(i+1:i+2,j+1:j+2).^2 + Dyu(i+1:i+2,j+1:j+2).^2 + epsilon);
        
        %-------------------------------------- update curvature terms
        Dx(i:i+2,j+1:j+2) = Dxu(i:i+2,j+1:j+2)./gx(i:i+2,j+1:j+2);
        Dy(i+1:i+2,j:j+2) = Dyu(i+1:i+2,j:j+2)./gy(i+1:i+2,j:j+2);
        kx(i:i+2,j:j+2) = diff(Dx(i:i+2,j:j+3),1,2);
        ky(i:i+2,j:j+2) = diff(Dy(i:i+3,j:j+2));
    end
end
end