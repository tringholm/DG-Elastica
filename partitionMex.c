// partitionMex propagates one timestep of the discrete gradient algorithm 
// on a border using a C implementation
// 
// Input:
// u       - current restored image
// Nx & Ny - image dimensions
// g       - noisy input greyscale image
// dt      - time step size
// a       - total variation regularization weight
// b       - curvature term regularization weight
// s       - fidelity term is computed in L^s norm
// epsilon - smoothing parameter
// tol     - tolerance for Brent-Dekker nonlinear equation solver
// K       - logical map with false values on pixels to be inpainted
// 
// Output:
// u       - updated border after one time step
// fcount  - total number of function used in timestep
// 
// Torbjørn Ringholm
// Email           : torbjorn.ringholm@ntnu.no
// Last updated    : 03/10/2017

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <matrix.h>
#include "mex.h"

#define u_in            prhs[0]
#define Nx_in           prhs[1]
#define Ny_in           prhs[2]
#define g_in            prhs[3]
#define dt_in           prhs[4]
#define a_in            prhs[5]
#define b_in            prhs[6]
#define s_in            prhs[7]
#define epsilon_in      prhs[8]
#define tol_in          prhs[9]
#define K_in            prhs[10]

// ---------------------------------------------- mex version of coordFxn
double coordFxnMex(double u,double u_old,double *u_l_in, double *Dxul_in, double *Dxuc_in,
        double *Dxur_in, double *Dxumt_in, double *Dxuml_in, double *Dyut_in, double*Dyuc_in,
        double *Dyub_in, double *Dyuml_in, double *Dyumr_in, double *gradabsmid_in,
        double *Dx_in, double*Dy_in, double E, double K, double g, int i, int j, int Ny, int Nx,
        double dt, double a, double b, double s, double epsilon);

// ---------------------------------------------- mex version of fzeroFast
void fzeroFastMex(double x,double u_old,double *u_l, double *Dxul, double *Dxuc,
        double *Dxur, double *Dxumt, double *Dxuml, double *Dyut, double*Dyuc,
        double *Dyub, double *Dyuml, double *Dyumr, double *gradabsmid,
        double *Dx, double*Dy, double E, double K, double g, int i, int j, int Ny, int Nx,
        double dt, double aa, double bb,
        double ss, double epsilon, double tol, double* out, double* fvalout, double *fcountout);

// ---------------------------------------------- mex version of dgstep
void dgstep(double **u, double *g, double *K, int Nx, int Ny, double dt, double a, double b,
        double s, double epsilon, double tol, double *fcountout);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // ------------------------------------------ initialize and get input
    double dt, a, b, s, epsilon, tol;
    double *u,*g, *u_out, *K, *fcountout;
    int Nx, Ny,i,j;
    
    Nx = *mxGetPr(Nx_in);
    Ny = *mxGetPr(Ny_in);
    
    dt = *mxGetPr(dt_in);
    a = *mxGetPr(a_in);
    b = *mxGetPr(b_in);
    s = *mxGetPr(s_in);
    epsilon = *mxGetPr(epsilon_in);
    tol = *mxGetPr(tol_in);
    fcountout = (double*)mxMalloc(sizeof(double));
    *fcountout = 0;
    
    
    u = mxGetPr(u_in);
    g = mxGetPr(g_in);
    K = mxGetPr(K_in);
    
    // ------------------------------------------ take one time step
    dgstep(&u, g, K, Nx, Ny, dt, a, b, s, epsilon, tol, fcountout);
    
    // ------------------------------------------ deliver output
    plhs[0] = mxCreateDoubleMatrix(Ny, Nx, mxREAL);
    u_out = mxGetPr(plhs[0]);
    
    for (i = 0; i < Nx; i++){
        for (j = 0; j < Ny; j++){
            u_out[j+ i*Ny] = u[j+2 + (i+1)*(Ny+4)];
        }
    }
    mxFree(u);
    plhs[1] = mxCreateDoubleScalar(*fcountout);
}

void dgstep(double **uptr, double *g, double *K, int Nx, int Ny, double dt, double a, double b,
        double s, double epsilon, double tol, double *fcountout){
    // ------------------------------------------ initialize variables
    double u_old, E, sqrteps, fcount, *u;
    int i, j, k, l, ind;
    double *out =        (double*)mxMalloc(sizeof(double));
    double *fvalout =    (double*)mxMalloc(sizeof(double));
    double* u2 =         (double*)mxCalloc((Ny+4)*(Nx+2), sizeof(double));
    double* Dxu =        (double*)mxCalloc((Ny+4)*(Nx+3), sizeof(double));
    double* Dyu =        (double*)mxCalloc((Ny+3)*(Nx+2), sizeof(double));
    double* Dxu2 =       (double*)mxCalloc((Ny+3)*(Nx+2), sizeof(double));
    double* Dyu2 =       (double*)mxCalloc((Ny+2)*(Nx+3), sizeof(double));
    double* gradabs =    (double*)mxCalloc((Ny+2)*(Nx+3), sizeof(double));
    double* gradabs2 =   (double*)mxCalloc((Ny+3)*(Nx+2), sizeof(double));
    double* Dx =         (double*)mxCalloc((Ny+2)*(Nx+3), sizeof(double));
    double* Dy =         (double*)mxCalloc((Ny+3)*(Nx+2), sizeof(double));
    double* D1 =         (double*)mxCalloc((Ny+2)*(Nx+2), sizeof(double));
    double* D2 =         (double*)mxCalloc((Ny+2)*(Nx+2), sizeof(double));
    double* gradabsmid = (double*)mxCalloc((Ny+2)*(Nx+2), sizeof(double));
    double u_l[9];
    double Dxul[9];
    double Dxuc[6];
    double Dxur[9];
    double Dxum[9];
    double Dxumt[6];
    double Dxuml[6];
    double Dyut[9];
    double Dyuc[6];
    double Dyub[9];
    double Dyum[9];
    double Dyuml[6];
    double Dyumr[6];
    double Dx_l[12];
    double Dy_l[12];
    double D1_l[9];
    double D2_l[9];
    double gradabsmid_l[9];
    
    u = *uptr;
    for (j = 1; j < Nx+1; j++){
        for (i = 0; i < Ny+4; i++){
            u2[i + j*(Ny+4)] = u[i + (j-1)*(Ny+4)];
        }
    }
    
    // ------------------------------------------ padding values
    sqrteps = sqrt(epsilon);
    for (i = 0; i < Ny+2; i++){
        gradabs[i] = sqrteps;
        gradabs[i+(Nx+2)*(Ny+2)] = sqrteps;
        gradabsmid[i] = sqrteps;
        gradabsmid[i+(Nx+1)*(Ny+2)] = sqrteps;
    }
    for (i = 0; i < Nx+3; i++){
        gradabs[i*(Ny+2)] = sqrteps;
        gradabs[(i+1)*(Ny+2) - 1] = sqrteps;
    }
    for (i = 0; i < Ny+3; i++){
        gradabs2[i] = sqrteps;
        gradabs2[i+(Nx+1)*(Ny+3)] = sqrteps;
    }
    for (i = 0; i < Nx+2; i++){
        gradabs2[i*(Ny+3)] = sqrteps;
        gradabs2[(i+1)*(Ny+3) - 1] = sqrteps;
        gradabsmid[i*(Ny+2)] = sqrteps;
        gradabsmid[(i+1)*(Ny+2) - 1] = sqrteps;
    }
    
    // ------------------------------------------ derivative approximations
    for (j = 2; j < Nx+1; j++){
        for (i = 0; i < Ny+4; i++){
            Dxu[i + j*(Ny+4)] = u[i + (j-1)*(Ny+4)] - u[i + (j-2)*(Ny+4)];
        }
    }
    for (j = 1; j < Nx+1; j++){
        for (i = 0; i < Ny+3; i++){
            Dyu[i + j*(Ny+3)] = u[i+1 + (j-1)*(Ny+4)] - u[i + (j-1)*(Ny+4)];
        }
    }
    
    // ------------------------------------------ derivative interpolations
    for (j = 1; j < Nx+1; j++){
        for (i = 0; i < Ny+3; i++){
            Dxu2[i + j*(Ny+3)] = 0.25*(Dxu[i+1 + j*(Ny+4)] + Dxu[i + j*(Ny+4)] + Dxu[i+1 + (j+1)*(Ny+4)] + Dxu[i + (j+1)*(Ny+4)]);
        }
    }
    for (j = 2; j < Nx+1; j++){
        for (i = 0; i < Ny+2; i++){
            Dyu2[i + j*(Ny+2)] = 0.25*(Dyu[i + j*(Ny+3)] + Dyu[i+1 + j*(Ny+3)] + Dyu[i + (j-1)*(Ny+3)] + Dyu[i+1 + (j-1)*(Ny+3)]);
        }
    }
    for (i = 0; i < Ny+2; i++){
        Dyu2[i + (Ny+2)] = 0.5*(Dyu[i + (Ny+3)] + Dyu[i+1 + (Ny+3)]);
    }
    for (i = 0; i < Ny+2; i++){
        Dyu2[i + (Nx+1)*(Ny+2)] = 0.5*(Dyu[i + Nx*(Ny+3)] + Dyu[i+1 + Nx*(Ny+3)]);
    }
    
    // ------------------------------------------ gradient moduli
    for (j = 1; j < Nx+2; j++){
        for (i = 0; i < Ny+2; i++){
            gradabs[i + j*(Ny+2)] = sqrt(Dxu[i+1 + j*(Ny+4)]*Dxu[i+1 + j*(Ny+4)] + Dyu2[i + j*(Ny+2)]*Dyu2[i + j*(Ny+2)] + epsilon);
        }
    }
    for (j = 1; j < Nx+1; j++){
        for (i = 0; i < Ny+3; i++){
            gradabs2[i + j*(Ny+3)] = sqrt(Dxu2[i + j*(Ny+3)]*Dxu2[i + j*(Ny+3)] + Dyu[i + j*(Ny+3)]*Dyu[i + j*(Ny+3)] + epsilon);
        }
    }
    
    // ------------------------------------------ curvature term 
    for (j = 0; j < Nx+3; j++){
        for (i = 0; i < Ny+2; i++){
            Dx[i + j*(Ny+2)] = Dxu[i+1 + j*(Ny+4)]/gradabs[i + j*(Ny+2)];
        }
    }
    for (j = 0; j < Nx+2; j++){
        for (i = 0; i < Ny+3; i++){
            Dy[i + j*(Ny+3)] = Dyu[i + j*(Ny+3)]/gradabs2[i + j*(Ny+3)];
        }
    }
    for (j = 0; j < Nx+2; j++){
        for (i = 0; i < Ny+2; i++){
            D1[i + j*(Ny+2)] = Dx[i + (j+1)*(Ny+2)] - Dx[i + j*(Ny+2)];
        }
    }
    for (j = 0; j < Nx+2; j++){
        for (i = 0; i < Ny+2; i++){
            D2[i + j*(Ny+2)] = Dy[i+1 + j*(Ny+3)] - Dy[i + j*(Ny+3)];
        }
    }
    
    
    for (j = 1; j < Nx+1; j++){
        for (i = 0; i < Ny+2; i++){
            gradabsmid[i + j*(Ny+2)] = sqrt(Dxu[i+1 + j*(Ny+4)]*Dxu[i+1 + j*(Ny+4)] + Dyu[i + j*(Ny+3)]*Dyu[i + j*(Ny+3)] + epsilon);
        }
    }
    
    fcount = 0;
    
    *uptr = u2;
    u = u2;
    // ------------------------------------------ main loop
    for (i = 0; i < 4; i++){ 
        for (j = 0; j < Nx; j++){ 
            E = 0;
            // ---------------------------------- extract local values
            for (k = 0; k < 3; k++){
                for (l = 0; l < 3; l++){
                    ind = l + 3*k;
                    u_l[ind] = u[i+l+1 + (j+k)*(Ny+4)];
                    Dxul[ind] = Dxu[i+l+1 + (j+k)*(Ny+4)];
                    Dxur[ind] = Dxu[i+l+1 + (j+1+k)*(Ny+4)];
                    Dxum[ind] = 0.25*(Dxul[ind] + Dxur[ind]);
                    Dyut[ind] = Dyu[i+l + (j+k)*(Ny+3)];
                    Dyub[ind] = Dyu[i+l+1 + (j+k)*(Ny+3)];
                    Dyum[ind] = 0.25*(Dyut[ind] + Dyub[ind]);
                    D1_l[ind] = D1[i+l + (j+k)*(Ny+2)];
                    D2_l[ind] = D2[i+l + (j+k)*(Ny+2)];
                    gradabsmid_l[ind] = gradabsmid[i+l + (j+k)*(Ny+2)];
                    // -------------------------- old energy in pixel
                    E = E + (a + b*(D1_l[ind] + D2_l[ind])*(D1_l[ind] + D2_l[ind]))* gradabsmid_l[ind];
                }
            }
            
            
            u_old = u_l[4];
            
            for (k = 0; k < 3; k++){
                for (l = 0; l < 2; l++){
                    ind = l + 2*k;
                    Dxumt[ind] = Dxum[ind + k];
                    Dxuml[ind] = Dxum[ind + 1 + k];
                    Dyuc[ind] = Dyu[i+l+1 + (j+k)*(Ny+3)];
                }
            }

            for (k = 0; k < 2; k++){
                for (l = 0; l < 3; l++){
                    ind = l + 3*k;
                    Dyuml[ind] = Dyum[ind];
                    Dyumr[ind] = Dyum[ind + 3];
                    Dxuc[ind] = Dxu[i+l+1 + (j+1+k)*(Ny+4)];
                }
            }
            
            for (k = 0; k < 4; k++){
                for (l = 0; l < 3; l++){
                    Dx_l[l + 3*k] = Dx[i+l + (j+k)*(Ny+2)];
                }
            }

            for (k = 0; k < 3; k++){
                for (l = 0; l < 4; l++){
                    Dy_l[l + 4*k] = Dy[i+l + (j+k)*(Ny+3)];
                }
            }

            // ---------------------------------- update pixel value
            fzeroFastMex(u_old+0.1, u_old,  u_l,  Dxul,  Dxuc, Dxur, Dxumt, Dxuml,
                    Dyut, Dyuc, Dyub, Dyuml, Dyumr, gradabsmid_l, Dx_l, Dy_l,
                    E, K[i + j*Ny], g[i + j*Ny], 1, j, Ny, Nx, dt,
                    a, b, s, epsilon, tol, out, fvalout, fcountout);

            u[i+2 + (j+1)*(Ny+4)] = *out;
            u_l[4] = *out;
            
            // ---------------------------------- update deriv approx
            if (j > 0){
                Dxu[i+2 + (j+1)*(Ny+4)] = u_l[4] - u_l[1];
            }
            if (j < Nx-1){
                Dxu[i+2 + (j+2)*(Ny+4)] = u_l[7] - u_l[4];
            }
            
            Dyu[i+1 + (j+1)*(Ny+3)] = u_l[4] - u_l[3];
            Dyu[i+2 + (j+1)*(Ny+3)] = u_l[5] - u_l[4];

            // ---------------------------------- update deriv interp
            for (k = 0; k < 3; k++){
                for (l = 1; l < 3; l++){
                    Dxu2[i+l + (j+k)*(Ny+3)] = 0.25*(Dxu[i+l + (j+k)*(Ny+4)] + Dxu[i+l+1 + (j+k)*(Ny+4)] + Dxu[i+l + (j+1+k)*(Ny+4)] + Dxu[i+l+1 + (j+k+1)*(Ny+4)]);
                }
            }
            for (k = 1; k < 3; k++){
                for (l = 0; l < 3; l++){
                    Dyu2[i+l + (j+k)*(Ny+2)] = 0.25*(Dyu[i+l + (j+k-1)*(Ny+3)] + Dyu[i+l + (j+k)*(Ny+3)] + Dyu[i+l+1 + (j-1+k)*(Ny+3)] + Dyu[i+l+1 + (j+k)*(Ny+3)]);
                }
            }
            
            if (j == 0){
                for (l = 0; l < 3; l++){
                    Dyu2[i+l + (j+1)*(Ny+2)] = 0.5*(Dyu[i+l + (j+1)*(Ny+3)] + Dyu[i+1+l + (j+1)*(Ny+3)]);
                }
            }
            else if (j == Nx-1){
                for (l = 0; l < 3; l++){
                    Dyu2[i+l + (j+2)*(Ny+2)] = 0.5*(Dyu[i+l + (j+1)*(Ny+3)] + Dyu[i+1+l + (j+1)*(Ny+3)]);
                }
            }
            
            // ---------------------------------- update gradient moduli
            // ---------------------------------- and curvature
            for (k = 1; k < 3; k++){
                for (l = 0; l < 3; l++){
                    gradabs[i+l + (j+k)*(Ny+2)] = sqrt(Dxu[i+l+1 + (j+k)*(Ny+4)]*Dxu[i+l+1 + (j+k)*(Ny+4)] + Dyu2[i+l + (j+k)*(Ny+2)]*Dyu2[i+l + (j+k)*(Ny+2)] + epsilon);
                    Dx[i+l + (j+k)*(Ny+2)] = Dxu[i+l+1 + (j+k)*(Ny+4)]/gradabs[i+l + (j+k)*(Ny+2)];
                }
            }
            for (k = 0; k < 3; k++){
                for (l = 1; l < 3; l++){
                    gradabs2[i+l + (j+k)*(Ny+3)] = sqrt(Dyu[i+l + (j+k)*(Ny+3)]*Dyu[i+l + (j+k)*(Ny+3)] + Dxu2[i+l + (j+k)*(Ny+3)]*Dxu2[i+l + (j+k)*(Ny+3)] + epsilon);
                    Dy[i+l + (j+k)*(Ny+3)] = Dyu[i+l + (j+k)*(Ny+3)]/gradabs2[i+l + (j+k)*(Ny+3)];
                }
            }
            
            for (k = 0; k < 3; k++){
                for (l = 0; l < 3; l++){
                    D1[i+l + (j+k)*(Ny+2)] = Dx[i+l + (j+1+k)*(Ny+2)] - Dx[i+l + (j+k)*(Ny+2)];
                    D2[i+l + (j+k)*(Ny+2)] = Dy[i+l+1 + (j+k)*(Ny+3)] - Dy[i+l + (j+k)*(Ny+3)];
                }
            }
            
            for (k = 1; k < 3; k++){
                for (l = 1; l < 3; l++){
                    gradabsmid[i+l + (j+k)*(Ny+2)] = sqrt(Dxu[i+l+1 + (j+k)*(Ny+4)]*Dxu[i+l+1 + (j+k)*(Ny+4)] + Dyu[i+l + (j+k)*(Ny+3)]*Dyu[i+l + (j+k)*(Ny+3)] + epsilon);
                }
            }
            
            
        }
    }
    
    // ------------------------------------------ deallocate variables
    mxFree(out);
    mxFree(fvalout);
    mxFree(Dxu);
    mxFree(Dyu);
    mxFree(Dxu2);
    mxFree(Dyu2);
    mxFree(gradabs);
    mxFree(gradabs2);
    mxFree(Dx);
    mxFree(Dy);
    mxFree(D1);
    mxFree(D2);
    mxFree(gradabsmid);
}


void fzeroFastMex(double x,double u_old,double *u_l, double *Dxul, double *Dxuc,
        double *Dxur, double *Dxumt, double *Dxuml, double *Dyut, double*Dyuc,
        double *Dyub, double *Dyuml, double *Dyumr, double *gradabsmid,
        double *Dx, double*Dy, double E, double K, double g, int i, int j, int Ny, int Nx,
        double dt, double aa, double bb,
        double ss, double epsilon, double tol, double* out, double* fvalout, double *fcountout){
    
    // ------------------------------------------ initialization
    double fx, dx, twosqrt, a, b, fa, fb, fc, c, d, e, m, toler, s, p, q, r, fval, fcount;
    int interval;
    
    fcount = 0;
    interval = 1;
    
    // ------------------------------------------ starting with interval
    if (interval){
        a = -0.1;
        b = 1.1;
        fa = coordFxnMex(a,u_old,u_l,Dxul,Dxuc,Dxur,Dxumt,Dxuml,Dyut,Dyuc,Dyub,Dyuml,Dyumr,gradabsmid,Dx,Dy,E,K,g,i,j,Ny,Nx,dt,aa,bb,ss,epsilon);
        fb = coordFxnMex(b,u_old,u_l,Dxul,Dxuc,Dxur,Dxumt,Dxuml,Dyut,Dyuc,Dyub,Dyuml,Dyumr,gradabsmid,Dx,Dy,E,K,g,i,j,Ny,Nx,dt,aa,bb,ss,epsilon);
        fcount = fcount + 2;
        if (fa == 0){
            fval = fa;
            *out = a;
            *fvalout = fval;
            *fcountout = *fcountout + fcount;
        }
        else if(fb ==0){
            fval = fb;
            *out = b;
            *fvalout = fval;
            *fcountout = *fcountout + fcount;
        }
    }
    // ------------------------------------------ no interval, initialize
    else{
        // -------------------------------------- evaluate at starting pt
        fx = coordFxnMex(x,u_old,u_l,Dxul,Dxuc,Dxur,Dxumt,Dxuml,Dyut,Dyuc,Dyub,Dyuml,Dyumr,gradabsmid,Dx,Dy,E,K,g,i,j,Ny,Nx,dt,aa,bb,ss,epsilon);
        
        fcount = fcount + 1;
        
        // -------------------------------------- set step sizes
        if (x != 0){
            dx = x/50;
        }
        else{
            dx = 1/50;
        }
        
        // -------------------------------------- find interval with zero
        twosqrt = sqrt(2);
        a = x; fa = fx; b = x; fb = fx;
        
        while ((fa > 0) == (fb > 0)){
            dx = twosqrt*dx;
            a = x - dx;
            fa = coordFxnMex(a,u_old,u_l,Dxul,Dxuc,Dxur,Dxumt,Dxuml,Dyut,Dyuc,Dyub,Dyuml,Dyumr,gradabsmid,Dx,Dy,E,K,g,i,j,Ny,Nx,dt,aa,bb,ss,epsilon);
            
            fcount = fcount + 1;
            
            if ((fa > 0) != (fb > 0)){
                break;
            }
            
            b = x + dx;
            fb = coordFxnMex(b,u_old,u_l,Dxul,Dxuc,Dxur,Dxumt,Dxuml,Dyut,Dyuc,Dyub,Dyuml,Dyumr,gradabsmid,Dx,Dy,E,K,g,i,j,Ny,Nx,dt,aa,bb,ss,epsilon);
            
            fcount = fcount + 1;
            
        }
    }

    fc = fb;
    // -------------------------------------- do main loop
    while (fb != 0 && a != b){
        // -------------------------------------- correct order
        if ((fb > 0) == (fc > 0)){
            c = a;  fc = fa;
            d = b - a;  e = d;
        }
        if (fabs(fc) < fabs(fb)){
            a = b;    b = c;    c = a;
            fa = fb;  fb = fc;  fc = fa;
        }
        
        // -------------------------------------- convergence test
        m = 0.5*(c - b);
        toler = 2.0*tol*((fabs(b)>1.0)?fabs(b):1.0);
        if ((fabs(m) <= toler) || (fb == 0.0)){
            break;
        }
        
        // -------------------------------------- bisection or interpol'n
        if ((fabs(e) < toler) || (fabs(fa) <= fabs(fb))){
            d = m;  e = m;
        }else{
            s = fb/fa;
            // ---------------------------------- linear or inverse quad'c
            if (a == c){
                p = 2.0*m*s;
                q = 1.0 - s;
            }else{
                q = fa/fc;
                r = fb/fc;
                p = s*(2.0*m*q*(q - r) - (b - a)*(r - 1.0));
                q = (q - 1.0)*(r - 1.0)*(s - 1.0);
            }
            if (p > 0){
                q = -q;
            }
            else{
                p = -p;
            }
            // ---------------------------------- goodness of interpol'n
            if ((2.0*p < 3.0*m*q - fabs(toler*q)) && (p < fabs(0.5*e*q))){
                e = d;  d = p/q;
            }
            else{
                d = m;  e = m;
            }
        }
        
        // -------------------------------------- prepare for next iter'n
        a = b;
        fa = fb;
        if (fabs(d) > toler){
            b = b + d;
        }else if (b > c){
            b = b - toler;
        }else{
            b = b + toler;
        }
        fb = coordFxnMex(b,u_old,u_l,Dxul,Dxuc,Dxur,Dxumt,Dxuml,Dyut,Dyuc,Dyub,Dyuml,Dyumr,gradabsmid,Dx,Dy,E,K,g,i,j,Ny,Nx,dt,aa,bb,ss,epsilon);
        
        fcount = fcount + 1;
    }
    
    fval = fb;
    
    // ------------------------------------------ set outputs
    *out = b;
    *fvalout = fval;
    *fcountout = *fcountout + fcount;
}

double coordFxnMex(double u,double u_old,double *u_l, double *Dxul, double *Dxuc,
        double *Dxur, double *Dxumt, double *Dxuml, double *Dyut, double*Dyuc,
        double *Dyub, double *Dyuml, double *Dyumr, double *gradabsmid,
        double *Dx, double*Dy, double E, double K, double g, int i, int j, int Ny, int Nx,
        double dt, double a, double b, double s, double epsilon){
    
    // ------------------------------------------ initialization
    double gamma, du, dum, dum2, out, out2, ff, ff2;
    int k,l;
    double Dxu2[6];
    double Dyu2[6];
    double gradabs[6];
    double gradabs2[6];
    double D[9];
    
    // ------------------------------------------ limit case
    gamma = 1.0/(u-u_old);
    if (u == u_old){
        return u + u_old - 2*g;
    }
    
    // ------------------------------------------ update differences
    if (j > 0){
        du = u - u_l[1];
        Dxul[4] = du;
        Dxuc[1] = du;
        dum = 0.25*(du + Dxul[1]);
        dum2 = 0.25*(du + Dxur[4]);
        Dxumt[1] = dum;
        Dxumt[3] = dum2;
        Dxuml[0] = dum;
        Dxuml[2] = dum2;
    }
    if (j < Nx-1){
        du = u_l[7] - u;
        Dxul[7] = du;
        gradabsmid[7] = sqrt(du*du  + Dyut[7]*Dyut[7] + epsilon);
        Dxuc[4] = du;
        dum = 0.25*(du + Dxul[4]);
        dum2 = 0.25*(du + Dxur[7]);
        Dxumt[3] = dum;
        Dxumt[5] = dum2;
        Dxuml[2] = dum;
        Dxuml[4] = dum2;
    }
    if (i > 0){
        du = u - u_l[3];
        Dyut[4] = du;
        Dyuc[2] = du;
        dum = 0.25*(du + Dyub[4]);
        dum2 = 0.25*(du + Dyut[3]);
        Dyuml[3] = dum2;
        Dyuml[4] = dum;
        Dyumr[0] = dum2;
        Dyumr[1] = dum;
    }
    if (i < Ny-1){
        du = u_l[5] - u;
        Dyut[5] = du;
        gradabsmid[5] = sqrt(Dxul[5]*Dxul[5]  + du*du + epsilon);
        Dyuc[3] = du;
        dum = 0.25*(du + Dyub[5]);
        dum2 = 0.25*(du + Dyut[4]);
        Dyuml[4] = dum2;
        Dyuml[5] = dum;
        Dyumr[1] = dum2;
        Dyumr[2] = dum;
    }
    
    // ------------------------------------------ update interpolations
    for (k = 0; k < 6; k++){
        Dxu2[k] = Dxumt[k] + Dxuml[k];
    }
    
    if(i == 0){
        for (k = 0; k < 3; k++){
            Dxu2[2*k] = 2.0*Dxumt[2*k+1];
        }
    }
    else if(i == Ny-1){
        for (k = 0; k < 3; k++){
            Dxu2[2*k+1] = 2.0*Dxumt[2*k+1];
        }
    }
    
    for (k = 0; k < 6; k++){
        Dyu2[k] = Dyuml[k] + Dyumr[k];
    }
    
    if(j == 0){
        for (k = 0; k < 3; k++){
            Dyu2[k] = 2.0*Dyuml[3 + k];
        }
    }
    else if(j == Nx-1){
        for (k = 0; k < 3; k++){
            Dyu2[3 + k] = 2.0*Dyuml[3 + k];
        }
    }
    
    // ------------------------------------------ update gradient moduli
    gradabsmid[4] = sqrt(Dxul[4]*Dxul[4]  + Dyut[4]*Dyut[4] + epsilon);
    for (k = 0; k < 6; k++){
        gradabs[k] = sqrt(Dxuc[k]*Dxuc[k] + Dyu2[k]*Dyu2[k] + epsilon);
    }
    for (k = 0; k < 6; k++){
        gradabs2[k] = sqrt(Dyuc[k]*Dyuc[k] + Dxu2[k]*Dxu2[k] + epsilon);
    }
    
    // ------------------------------------------ update curvature
    for (k = 0; k < 6; k++){
        Dx[3+k] = Dxuc[k]/gradabs[k];
    }
    for (k = 0; k < 3; k++){
        Dy[1+4*k] = Dyuc[2*k]/gradabs2[2*k];
        Dy[2+4*k] = Dyuc[2*k+1]/gradabs2[2*k+1];
    }
    for (k = 0; k < 3; k++){
        for(l = 0; l < 3; l++){
            D[3*k+l] = Dx[3*k+l+3] - Dx[3*k+l] + Dy[1+l+4*k] - Dy[l+4*k];
        }
    }
    
    // ------------------------------------------ calculate output
    out = 0;
    out2 = 0;
    
    for (k = 0; k < 9; k++){
        out += gradabsmid[k];
        out2 += D[k]*D[k]*gradabsmid[k];
    }
    out *= a;
    out2 *= b;
    out += out2;
    
    if (s == 2){
        ff = u-g;
        ff2 = u_old-g;
        out += - E + 0.5*K*(ff*ff - ff2*ff2);
    }else{
        ff = fabs(u-g);
        ff2 = fabs(u_old-g);
        out += - E + K*(ff- ff2);
    }
    out *= gamma;
    out += (u-u_old)/dt;
    
    
    return out;
}
