function [b,fval,fcount] = fzeroFast(u,tol,u_old,u_l,Dxul,Dxuc,Dxur,Dxumt,Dxuml,Dyut,Dyuc,Dyub,Dyuml,Dyumr,gm,Dx,Dy,E,K,g,i,j,Ny,Nx,dt,aa,bb,ss,epsilon)

% fzeroFast is a lightweight version of MATLAB's fzero, using the
% Brent-Dekker algorithm
%
% Input:
% u         - starting guess for current pixel
% tol       - tolerance for stopping criterion
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
% aa        - total variation regularization weight
% bb        - curvature term regularization weight
% ss        - fidelity term is computed in L^s norm
% epsilon   - smoothing parameter
%
% Output:
% b         - point of zero crossing
% fval      - value of objective at b
% fcount       - number of function interations
%
% Torbjørn Ringholm
% Email           : torbjorn.ringholm@ntnu.no
% Last updated    : 03/10/2017

%---------------------------------------------- initialization
fcount = 0;

%---------------------------------------------- evaluate at starting point
fx = coordFxn(u,u_old,u_l,Dxul,Dxuc,Dxur,Dxumt,Dxuml,Dyut,Dyuc,Dyub,Dyuml,Dyumr,gm,Dx,Dy,E,K,g,i,j,Ny,Nx,dt,aa,bb,ss,epsilon);
fcount = fcount + 1;

%---------------------------------------------- set step sizes
if u ~= 0
    dx = u/50;
else
    dx = 1/50;
end

%---------------------------------------------- find interval with zero
twosqrt = sqrt(2);
a = u; fa = fx; b = u; fb = fx;
while (fa > 0) == (fb > 0)
    dx = twosqrt*dx;
    a = u - dx;  
    fa = coordFxn(a,u_old,u_l,Dxul,Dxuc,Dxur,Dxumt,Dxuml,Dyut,Dyuc,Dyub,Dyuml,Dyumr,gm,Dx,Dy,E,K,g,i,j,Ny,Nx,dt,aa,bb,ss,epsilon);
    fcount = fcount + 1;
    
    if (fa > 0) ~= (fb > 0) 
        break
    end
    
    b = u + dx;  
    fb = coordFxn(b,u_old,u_l,Dxul,Dxuc,Dxur,Dxumt,Dxuml,Dyut,Dyuc,Dyub,Dyuml,Dyumr,gm,Dx,Dy,E,K,g,i,j,Ny,Nx,dt,aa,bb,ss,epsilon);
    fcount = fcount + 1;
end 

fc = fb;
%---------------------------------------------- with interval, do main loop
while fb ~= 0 && a ~= b
    %------------------------------------------ correct order
    if (fb > 0) == (fc > 0)
        c = a;  fc = fa;
        d = b - a;  e = d;
    end
    if abs(fc) < abs(fb)
        a = b;    b = c;    c = a;
        fa = fb;  fb = fc;  fc = fa;
    end
    
    %------------------------------------------ convergence test
    m = 0.5*(c - b);
    toler = 2.0*tol*max(abs(b),1.0);
    if (abs(m) <= toler) || (fb == 0.0) 
        break
    end
    
    %------------------------------------------ bisection or interpolation
    if (abs(e) < toler) || (abs(fa) <= abs(fb))
        d = m;  e = m;
    else
        s = fb/fa;
        %-------------------------------------- linear or inverse quadratic
        if (a == c)
            p = 2.0*m*s;
            q = 1.0 - s;
        else
            q = fa/fc;
            r = fb/fc;
            p = s*(2.0*m*q*(q - r) - (b - a)*(r - 1.0));
            q = (q - 1.0)*(r - 1.0)*(s - 1.0);
        end
        if p > 0 
            q = -q;
        else
            p = -p; 
        end
        %-------------------------------------- goodness of interpolation
        if (2.0*p < 3.0*m*q - abs(toler*q)) && (p < abs(0.5*e*q))
            e = d;  d = p/q;
        else
            d = m;  e = m;
        end
    end 
    
    %------------------------------------------ prepare for next iteration
    a = b;
    fa = fb;
    if abs(d) > toler 
        b = b + d;
    elseif b > c 
        b = b - toler;
    else
        b = b + toler;
    end
    fb = coordFxn(b,u_old,u_l,Dxul,Dxuc,Dxur,Dxumt,Dxuml,Dyut,Dyuc,Dyub,Dyuml,Dyumr,gm,Dx,Dy,E,K,g,i,j,Ny,Nx,dt,aa,bb,ss,epsilon);
    fcount = fcount + 1;
end 

%---------------------------------------------- finalize
fval = fb;

end