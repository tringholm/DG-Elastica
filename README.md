EulerElastica: Euler’s elastica-based image denoising and inpainting in MATLAB.

EulerElastica is a MATLAB based image analysis tool for denoising and/or inpainting damaged images using Euler’s elastica as a regularisation prior in a variational image analysis setting. The algorithm used for minimizing the resulting energy is the Discrete Gradient algorithm outlined in [1].

To get started, open the demo.m file and read the commented code there.

The package contains 12 MATLAB files and 5 C files, with the following structure:

- eulerElastica.m is the main file, and the only one you need to use to apply the algorithm
- defaultOptions.m supplies an option struct which may be useful to tweak parameters
- demo.m is a demonstration file showing how to use the algorithm with examples
- eulerElasticaMatlab/Mex/MexPara.m are lower level files executing the algorithm in MATLAB, MEX and Parallel versions, respectively
- coordFxn.m, dgstep.m, energyFxn.m, fzeroFast.m, gradCurv.m, and gradTV.m are all auxiliary functions used mostly in eulerElasticaMatlab.m
- dgstepMex/MexPara.c, gradCurvMex/MexPara.c and partitionMex.c are C functions for speeding up critical parts of the code in a MEX environment.

This is the first release, so if you have comments and suggestions for improvements, please contact me and I will try to accommodate them.
