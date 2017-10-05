function out = coordFxn(u,u_old,u_l,Dxul,Dxuc,Dxur,Dxumt,Dxuml,Dyut,Dyuc,Dyub,Dyuml,Dyumr,gm,Dx,Dy,E,K,g,i,j,Ny,Nx,dt,a,b,s,epsilon)

% coordFxn evaluates the function whose zero is to be found when taking a
% time step
%
% Input:
% u         - current pixel
% u_old     - pixel from previous time step
% u_l       - 3x3 matrix of surrounding pixels
% 5 x Dx*** - precomputed values of x derivatives
% 5 x Dy*** - precomputed values of y derivatives
% gm        - precomputed values of gradient modulus
% Dx & Dy   - precomputed values of curvature components
% E         - energy from previous time step
% K         - boolean with false value if inpainting pixel
% g         - input greyscale image pixel
% i & j     - indices of current pixel
% Ny & Nx   - dimensions of image
% dt        - time step size
% a         - total variation regularization weight
% b         - curvature term regularization weight
% s         - fidelity term is computed in L^s norm
% epsilon   - smoothing parameter
%
% Output:
% out       - value of LHS - RHS in Itoh-Abe coordinate update
%
% Torbjørn Ringholm
% Email           : torbjorn.ringholm@ntnu.no
% Last updated    : 03/10/2017

%---------------------------------------------- limit case
if u == u_old
    out = u + u_old - 2*g;
    return
end
gamma = 1/(u-u_old);

%---------------------------------------------- update differences 
if j > 1
    du = (u - u_l(2,1));
    Dxul(2,2) = du;
    Dxuc(2,1) = du;
    dum = 0.25*(du + Dxul(2,1));
    dum2 = 0.25*(du + Dxur(2,2));
    Dxumt(2,1) = dum;
    Dxuml(1,1) = dum;
    Dxumt(2,2) = dum2;
    Dxuml(1,2) = dum2;
end
if j < Nx
    du = (u_l(2,3) - u);
    Dxul(2,3) = du;
    gm(2,3) = realsqrt(du^2  + Dyut(2,3)^2 + epsilon);
    Dxuc(2,2) = du;
    dum = 0.25*(du + Dxul(2,2));
    dum2 = 0.25*(du + Dxur(2,3));
    Dxumt(2,2) = dum;
    Dxuml(1,2) = dum;
    Dxumt(2,3) = dum2;
    Dxuml(1,3) = dum2;
end
if i > 1
    du = (u - u_l(1,2));
    Dyut(2,2) = du;
    Dyuc(1,2) = du;
    dum = 0.25*(du + Dyub(2,2));
    dum2 = 0.25*(du + Dyut(1,2));
    Dyuml(2,2) = dum;
    Dyumr(2,1) = dum;
    Dyuml(1,2) = dum2;
    Dyumr(1,1) = dum2;
end
if i < Ny
    du = (u_l(3,2) - u);
    Dyut(3,2) = du;
    gm(3,2) = realsqrt(Dxul(3,2)^2  + du^2 + epsilon);
    Dyuc(2,2) = du;
    dum = 0.25*(du + Dyub(3,2));
    dum2 = 0.25*(du + Dyut(2,2));
    Dyuml(3,2) = dum;
    Dyumr(3,1) = dum;
    Dyuml(2,2) = dum2;
    Dyumr(2,1) = dum2;
end

%---------------------------------------------- update interpolations
Dxu2 = Dxumt + Dxuml;
if i == 1
    Dxu2(1,:) = 2*Dxumt(2,:);
elseif i == Ny
    Dxu2(2,:) = 2*Dxumt(2,:);
end
Dyu2 = Dyuml + Dyumr;
if j == 1
    Dyu2(:,1) = 2*Dyuml(:,2);
elseif j == Nx
    Dyu2(:,2) = 2*Dyuml(:,2);
end

%---------------------------------------------- update gradient moduli
gm(2,2) = realsqrt(Dxul(2,2)^2  + Dyut(2,2)^2 + epsilon);
gx = realsqrt(Dxuc.*Dxuc + Dyu2.*Dyu2 + epsilon);
gy = realsqrt(Dxu2.*Dxu2 + Dyuc.*Dyuc + epsilon);

%---------------------------------------------- update curvature
Dx(:,2:3) = Dxuc./gx;
Dy(2:3,:) = Dyuc./gy;
D = diff(Dx,1,2) + diff(Dy);

%---------------------------------------------- calculate output
out = gamma*sum(sum((a + b*D.^2).*gm));
out = out - gamma*E + K*(gamma/s*abs(u-g).^s - gamma/s*abs(u_old-g).^s);
out = out + (u-u_old)/dt;
end