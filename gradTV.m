function gradf = gradTV(u,epsilon)

% gradCurv computes the gradient of the total variation term, used for
% adaptivity
%
% Input:
% u       - current restored image
% epsilon - smoothing parameter
%
% Output:
% gradf   - gradient in the shape of a matrix
%
% Torbjørn Ringholm
% Email           : torbjorn.ringholm@ntnu.no
% Last updated    : 03/10/2017

%---------------------------------------------- derivative approximations
[Ny,Nx] = size(u);
Dxu = zeros(Ny,Nx+1);  
Dyu = zeros(Ny+1,Nx); 
Dxu(:,2:end-1) = diff(u,1,2);
Dyu(2:end-1,:) = diff(u);

%---------------------------------------------- gradient modulus
g = realsqrt(Dxu(:,1:end-1).^2 + Dyu(1:end-1,:).^2 + epsilon);

%---------------------------------------------- deriv of gradient modulus
gradf = (Dxu(:,1:end-1) + Dyu(1:end-1,:))./g;
gradf(:,1:end-1) = gradf(:,1:end-1) - Dxu(:,2:end-1)./g(:,2:end);
gradf(1:end-1,:) = gradf(1:end-1,:) - Dyu(2:end-1,:)./g(2:end,:);
end