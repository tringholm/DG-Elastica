// gradCurvMexPara computes the gradient of the discretized curvature term, used
// on a block, for adaptivity, in a C implementation
// 
// Input:
// u       - current restored image
// epsilon - smoothing parameter
// Nx & Ny - block dimensions
// Nytrue  - image y dimension
// lowind  - lower y index
// 
// Output:
// gradf   - gradient in the shape of a matrix
// 
// Torbjørn Ringholm
// Email           : torbjorn.ringholm@ntnu.no
// Last updated    : 03/10/2017

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <matrix.h>
#include "mex.h"

#define u_in            prhs[0]   //double *
#define epsilon_in      prhs[1]   //double
#define Nx_in           prhs[2]  //double
#define Ny_in           prhs[3]  //double
#define Nytrue_in       prhs[4]  //double
#define lowind_in       prhs[5]  //double

// ---------------------------------------------- mex version of gradCurv
void gradcurv2(double *u, double *gradf, double epsilon, int Nx, int Ny, int Nytrue, int lowind);

// ---------------------------------------------- helper function
double dux(int i, int j, int k, int l, int Nx);

// ---------------------------------------------- helper function
double duy(int i, int j, int k, int l, int Ny);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // ------------------------------------------ initialize and get input
    double epsilon;
    double *u, *gradf, *gradf_out;
    int Nx, Ny, Nytrue, lowind, i, j;
    
    Nx = *mxGetPr(Nx_in);
    Ny = *mxGetPr(Ny_in);
    Nytrue = *mxGetPr(Nytrue_in);
    lowind = *mxGetPr(lowind_in);
    
    epsilon = *mxGetPr(epsilon_in);
    gradf = (double*)mxCalloc(Ny*Nx,sizeof(double));
    
    u = mxGetPr(u_in);
    
    // ------------------------------------------ calculate gradient
    gradcurv2(u, gradf, epsilon, Nx, Ny, Nytrue, lowind);
    
    // ------------------------------------------ deliver output
    plhs[0] = mxCreateDoubleMatrix(Ny, Nx, mxREAL);
    gradf_out = mxGetPr(plhs[0]);
    for (i = 0; i < Nx; i++){
        for (j = 0; j < Ny; j++){
            gradf_out[j+ i*Ny] = gradf[j+ i*Ny];
        }
    }
}

void gradcurv2(double *u, double *gradf, double epsilon, int Nx, int Ny, int Nytrue, int lowind){
    
    // ------------------------------------------ initialize variables
    double* Dxu =        (double*)mxCalloc(Ny*(Nx+1), sizeof(double));
    double* Dyu =        (double*)mxCalloc((Ny+1)*Nx, sizeof(double));
    double* Dxu2 =       (double*)mxCalloc((Ny+1)*Nx, sizeof(double));
    double* Dyu2 =       (double*)mxCalloc(Ny*(Nx+1), sizeof(double));
    double* gx =         (double*)mxCalloc(Ny*(Nx+1), sizeof(double));
    double* gy =         (double*)mxCalloc((Ny+1)*Nx, sizeof(double));
    double* Dx =         (double*)mxCalloc(Ny*(Nx+1), sizeof(double));
    double* Dy =         (double*)mxCalloc((Ny+1)*Nx, sizeof(double));
    double* kx =         (double*)mxCalloc(Ny*Nx, sizeof(double));
    double* ky =         (double*)mxCalloc(Ny*Nx, sizeof(double));
    double* ksum =       (double*)mxCalloc(Ny*Nx, sizeof(double));
    double* g =          (double*)mxCalloc(Ny*Nx, sizeof(double));
    double* ginv =       (double*)mxCalloc(Ny*Nx, sizeof(double));
    
    
    int i,j,k,l,k_low,k_high,l_low,l_high, iglob, kglob,ind, ind2, ind3, ind4;
    double duxp, duxm, duyp, duym, dux2p, dux2m, duy2p, duy2m, dgxp, dgxm, dgyp, dgym, dgu, kxu, kyu;

    // ------------------------------------------ derivative approximations
    for (j = 1; j < Nx; j++){
        for (i = 0; i < Ny; i++){
            Dxu[i + j*Ny] = u[i + j*Ny] - u[i + (j-1)*Ny];
        }
    }
    for (j = 0; j < Nx; j++){
        for (i = 1; i < Ny; i++){
            Dyu[i + j*(Ny+1)] = u[i + j*Ny] - u[i-1 + j*Ny];
        }
    }
    
    // ------------------------------------------ derivative interpolations
    for (j = 0; j < Nx; j++){
        for (i = 1; i < Ny; i++){
            Dxu2[i + j*(Ny+1)] = 0.25*(Dxu[i + j*Ny] + Dxu[i-1 + j*Ny] + Dxu[i + (j+1)*Ny] + Dxu[i-1 + (j+1)*Ny]);
        }
    }
    for (j = 0; j < Nx; j++){
        Dxu2[j*(Ny+1)] = 0.5*(Dxu[j*Ny] + Dxu[(j+1)*Ny]);
    }
    for (j = 0; j < Nx; j++){
        Dxu2[Ny + j*(Ny+1)] = 0.5*(Dxu[Ny-1 + j*Ny] + Dxu[Ny-1 + (j+1)*Ny]);
    }
    
    for (j = 1; j < Nx; j++){
        for (i = 0; i < Ny; i++){
            Dyu2[i + j*Ny] = 0.25*(Dyu[i + j*(Ny+1)] + Dyu[i+1 + j*(Ny+1)] + Dyu[i + (j-1)*(Ny+1)] + Dyu[i+1 + (j-1)*(Ny+1)]);
        }
    }
    for (i = 0; i < Ny; i++){
        Dyu2[i] = 0.5*(Dyu[i] + Dyu[i+1]);
    }
    for (i = 0; i < Ny; i++){
        Dyu2[i + Nx*Ny] = 0.5*(Dyu[i + (Nx-1)*(Ny+1)] + Dyu[i+1 + (Nx-1)*(Ny+1)]);
    } 
    
    
    // ------------------------------------------ gradient moduli
    for (j = 0; j < Nx+1; j++){
        for (i = 0; i < Ny; i++){
            gx[i + j*Ny] = 1.0/sqrt(Dxu[i + j*Ny]*Dxu[i + j*Ny] + Dyu2[i + j*Ny]*Dyu2[i + j*Ny] + epsilon);
        }
    }
    for (j = 0; j < Nx; j++){
        for (i = 0; i < Ny+1; i++){
            gy[i + j*(Ny+1)] = 1.0/sqrt(Dxu2[i + j*(Ny+1)]*Dxu2[i + j*(Ny+1)] + Dyu[i + j*(Ny+1)]*Dyu[i + j*(Ny+1)] + epsilon);
        }
    }
    for (j = 0; j < Nx; j++){
        for (i = 0; i < Ny; i++){
            g[i + j*Ny] = sqrt(Dxu[i + j*Ny]*Dxu[i + j*Ny] + Dyu[i + j*(Ny+1)]*Dyu[i + j*(Ny+1)] + epsilon);
            ginv[i + j*Ny] = 1.0/g[i + j*Ny];
        }
    }
    
    // ------------------------------------------ curvature term 
    for (j = 0; j < Nx+1; j++){
        for (i = 0; i < Ny; i++){
            Dx[i + j*Ny] = Dxu[i + j*Ny]*gx[i + j*Ny];
        }
    }
    
    for (j = 0; j < Nx; j++){
        for (i = 0; i < Ny+1; i++){
            Dy[i + j*(Ny+1)] = Dyu[i + j*(Ny+1)]*gy[i + j*(Ny+1)];
        }
    }
    
    for (j = 0; j < Nx; j++){
        for (i = 0; i < Ny; i++){
            kx[i + j*Ny] = Dx[i + (j+1)*Ny] - Dx[i + j*Ny];
        }
    }
    
    for (j = 0; j < Nx; j++){
        for (i = 0; i < Ny; i++){
            ky[i + j*Ny] = Dy[i+1 + j*(Ny+1)] - Dy[i + j*(Ny+1)];
        }
    }
    
    for (j = 0; j < Nx; j++){
        for (i = 0; i < Ny; i++){
            ksum[i + j*Ny] = kx[i + j*Ny] + ky[i + j*Ny];
        }
    }
    

    // ------------------------------------------ index-by-index gradient
    for (j = 0; j < Nx; j++){
        l_low = j-1;
        if (l_low < 0){
            l_low = 0;
        }
        l_high = j+2;
        if (l_high > Nx){
            l_high = Nx;
        }
        for (i = 0; i < Ny; i++){
            k_low = i-1;
            if (k_low < 0){
                k_low = 0;
            }
            k_high = i+2;
            if (k_high > Ny){
                k_high = Ny;
            }
            iglob = i + lowind;
            // -------------------------------------- use certain indices
            for (k = k_low; k < k_high; k++){
                kglob = k + lowind;
                for (l = l_low; l < l_high; l++){
                    ind = k + l*Ny;
                    ind2 = ind + Ny;
                    ind3 = ind + l;
                    ind4 = ind + l + 1;
                    
                    // ------------------------------ derivs of differences
                    duxp = dux(iglob,j,kglob,l+1,Nx);
                    duxm = dux(iglob,j,kglob,l,Nx);
                    duyp = duy(iglob,j,kglob+1,l,Nytrue);
                    duym = duy(iglob,j,kglob,l,Nytrue);
                    
                    // ------------------------------ derivs of interpols
                    dux2p = 0.25*(duxm + dux(iglob,j,kglob+1,l,Nx) + duxp + dux(iglob,j,kglob+1,l+1,Nx));
                    dux2m = 0.25*(dux(iglob,j,kglob-1,l,Nx) + duxm + dux(iglob,j,kglob-1,l+1,Nx) + duxp);
                    duy2p = 0.25*(duym + duyp + duy(iglob,j,kglob,l+1,Nytrue) + duy(iglob,j,kglob+1,l+1,Nytrue));
                    duy2m = 0.25*(duy(iglob,j,kglob,l-1,Nytrue) + duy(iglob,j,kglob+1,l-1,Nytrue) + duym + duyp);
                    
                    if (kglob == 0){
                        dux2m = 0.5*(duxm + duxp);
                        duym = 0;
                    }
                    else if (kglob == Nytrue-1){
                        dux2p = 0.5*(duxm + duxp);
                        duyp = 0;
                    }
                    
                    if (l == 0){
                        duy2m = 0.5*(duym + duyp);
                        duxm = 0;
                    }
                    else if (l == Nx-1){
                        duy2p = 0.5*(duym + duyp);
                        duxp = 0;
                    }
                    
                    // ------------------------------ derivs of moduli
                    dgxp = (Dxu[ind2]*duxp + Dyu2[ind2]*duy2p)*gx[ind2];
                    dgxm = (Dxu[ind]*duxm + Dyu2[ind]*duy2m)*gx[ind];
                    dgyp = (Dyu[ind4]*duyp + Dxu2[ind4]*dux2p)*gy[ind4];
                    dgym = (Dyu[ind3]*duym + Dxu2[ind3]*dux2m)*gy[ind3];
                    dgu = (Dxu[ind]*duxm + Dyu[ind3]*duym)*ginv[ind];
                    
                    // ------------------------------ derivs of curvatures
                    kxu = (duxp - Dx[ind2]*dgxp)*gx[ind2] - (duxm - Dx[ind]*dgxm)*gx[ind];
                    kyu = (duyp - Dy[ind4]*dgyp)*gy[ind4] - (duym - Dy[ind3]*dgym)*gy[ind3];
                    
                    // ------------------------------ deriv of curvature
                    gradf[i + j*Ny] = gradf[i + j*Ny] + 2*ksum[ind]*(kxu + kyu)*g[ind] + ksum[ind]*ksum[ind]*dgu;
                }
                
            }
            
        }
    }


    // ------------------------------ deallocate variables
    mxFree(Dxu);
    mxFree(Dyu);
    mxFree(Dxu2);
    mxFree(Dyu2);
    mxFree(gx);
    mxFree(gy);
    mxFree(Dx);
    mxFree(Dy);
    mxFree(kx);
    mxFree(ky);
    mxFree(ksum);
    mxFree(g);
    mxFree(ginv);
    
    
    }

double dux(int i, int j, int k, int l, int Nx){
    // ------------------------------ derivative of x differences
    return (k == i)*((l == j && l > 0) - (j == l - 1 && l < Nx ));
}

double duy(int i, int j, int k, int l, int Ny){
    // ------------------------------ derivative of y differences
    return (l == j)*((i == k && k > 0) - (i == k-1 && k < Ny));
}
 
 
