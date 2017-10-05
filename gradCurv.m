function gradf = gradCurv(u,epsilon)
% 
% gradCurv computes the gradient of the discretized curvature term, used
% for adaptivity
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

%---------------------------------------------- initialize gradient
gradf = zeros(size(u));

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

%---------------------------------------------- index-by-index gradient
for i = 1:Ny
    indk = max(1,i-1):min(Ny,i+1);
    for j = 1:Nx
        indl = max(1,j-1):min(Nx,j+1);
        %-------------------------------------- use certain indices
        for k = indk
            for l = indl
                %------------------------------ derivs of differences
                duxp = dux(i,j,k,l+1,Nx);
                duxm = dux(i,j,k,l,Nx);
                duyp = duy(i,j,k+1,l,Ny);
                duym = duy(i,j,k,l,Ny);
                
                %------------------------------ derivs of interpolations
                dux2p = 0.25*(duxm + dux(i,j,k+1,l,Nx) + duxp + dux(i,j,k+1,l+1,Nx));
                dux2m = 0.25*(dux(i,j,k-1,l,Nx) + duxm + dux(i,j,k-1,l+1,Nx) + duxp);
                duy2p = 0.25*(duym+ duyp + duy(i,j,k,l+1,Ny) + duy(i,j,k+1,l+1,Ny));
                duy2m = 0.25*(duy(i,j,k,l-1,Ny) + duy(i,j,k+1,l-1,Ny) + duym + duyp);
                if k == 1
                    dux2m = 0.5*(duxm + duxp);
                    duym = 0;
                elseif k == Ny
                    dux2p = 0.5*(duxm + duxp);
                    duyp = 0;
                end
                
                if l == 1
                    duy2m = 0.5*(duym + duyp);
                    duxm = 0;
                elseif l == Nx
                    duy2p = 0.5*(duym + duyp);
                    duxp = 0;
                end
                
                %------------------------------ derivs of gradient moduli
                dgxp = (Dxu(k,l+1)*duxp + Dyu2(k,l+1)*duy2p)/gx(k,l+1);
                dgxm = (Dxu(k,l)*duxm + Dyu2(k,l)*duy2m)/gx(k,l);
                dgyp = (Dyu(k+1,l)*duyp + Dxu2(k+1,l)*dux2p)/gy(k+1,l);
                dgym = (Dyu(k,l)*duym + Dxu2(k,l)*dux2m)/gy(k,l);
                dgu = (Dxu(k,l)*duxm + Dyu(k,l)*duym)/gm(k,l);

                %------------------------------ derivs of curvature terms
                kxu = (duxp/gx(k,l+1) - Dx(k,l+1)/gx(k,l+1)*dgxp - duxm/gx(k,l) + Dx(k,l)/gx(k,l)*dgxm);
                kyu = (duyp/gy(k+1,l) - Dy(k+1,l)/gy(k+1,l)*dgyp - duym/gy(k,l) + Dy(k,l)/gy(k,l)*dgym);
                
                %------------------------------ deriv of curvature
                gradf(i,j) = gradf(i,j) + 2*(kx(k,l)+ky(k,l))*(kxu + kyu)*gm(k,l) + (kx(k,l)+ky(k,l))^2*dgu;

            end
        end
    end
end
end

function d = dux(i,j,k,l,Nx)
%------------------------------ derivative of x difference approximation
d = ((k == i && l == j && l > 1) - (k == i && j == l - 1 && l < Nx +1 ));
                
end

function d = duy(i,j,k,l,Ny)
%------------------------------ derivative of y difference approximation
d = ((i == k && k > 1 && l == j) - (i == k-1 && k < Ny +1 && l == j));
                
end
