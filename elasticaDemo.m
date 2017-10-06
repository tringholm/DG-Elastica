%
% elasticaDemo() serves as an introduction to the use of the algorithm, 
% showcasing basic examples of use in denoising and inpainting, as well as 
% some of the customization options in the options struct. 
%
% I recommend that you set a breakpoint at the beginning of the file and 
% step through it to get an overview of how to use this package.
%
% Torbjørn Ringholm
% Email           : torbjorn.ringholm@ntnu.no
% Last updated    : 06/10/2017

clear all
close all

%--------------------------------------------------------------------------
%----------------------- Basic L1 denoising example -----------------------
%--------------------------------------------------------------------------
%
% This example shows you how to do basic denoising using Euler's elastica.
% First, a greyscale image is read, then rescaled to take values in [0,1]
% before impulse noise is added. Note that for regular use, the noise will 
% already be in the input image. 
%
% Next, the regularization parameters are chosen. Typically, for denoising,
% a choice of 1 > a > b > 0.1 is reasonable. Also note the choice of norm 
% for the fidelity term. Taking s = 1 is best for impulse noise, while 
% s = 2 is best for Gaussian noise. 
%
% Finally, the algorithm is run.

%---------------------------------------------- input a grayscale image 
originalImage = imread('cameraman.tif');

%---------------------------------------------- rescale image to [0,1]
originalImage = double(originalImage)./max(max(double(originalImage)));

%---------------------------------------------- add impulse noise
rng(42); g = imnoise(originalImage,'salt & pepper',0.15);

%---------------------------------------------- set parameters
a = 0.5;  % Intensity of TV term
b = 0.1;  % Intensity of curvature term
s = 1;    % Fidelity term in L^s norm

%---------------------------------------------- run denoising algorithm
u_basic = eulerElastica(g,a,b,s);

%---------------------------------------------- show original image
figure, imagesc(originalImage); colormap(gray); pause(0.01)

close all






%--------------------------------------------------------------------------
%------------------ L1 denoising example with options ---------------------
%--------------------------------------------------------------------------
%
% This example differs from the above in that an option struct is used for
% tuning and altering outputs of the algorithm. For a complete list of
% options, consult defaultOptions.m. Note that the option struct is passed
% as a variable to eulerElastica.m.
%
% Here, we tell the algorithm to save the output image as 'demo.png', to
% show details about the iterations in the MATLAB terminal, and to tighten
% the convergence tolerances for both the DG algorithm itself
% (residualTol), and for the scalar subproblems (scalarTol).

%---------------------------------------------- set options
options = defaultOptions(b,s);
options.saveOutput = 1;
options.outputName = 'demo.png';
options.showDetails = 1;
options.residualTol = 1E-7;
options.scalarTol = 1E-7;
%---------------------------------------------- run denoising algorithm
u_options = eulerElastica(g,a,b,s,options);

%---------------------------------------------- show original image
figure, imagesc(u_options); colormap(gray); pause(0.01)
figure, imagesc(u_basic); colormap(gray); pause(0.01)

close all





%--------------------------------------------------------------------------
%-------------------- Parallel inpainting example  ------------------------
%--------------------------------------------------------------------------
%
% This example shows you how to do inpainting using Euler's elastica.
% First, a greyscale image is read, then rescaled. After this, 75% of the 
% pixels are given the value 0, denoting data loss. The data loss is marked
% in the pixel mask variable K, which has the values 1 and 0, representing
% no data loss and data loss, respectively. This defines the inpainting
% domain, and is passed as an input to the algorithm.
%
% Next, the regularization parameters are chosen. This can be tricky with
% inpainting, but a rule of thumb is that the more data loss occurs, the
% smaller a and b should be chosen, since this implies that fidelity with 
% the non-lossy pixels is emphasized. For inpainting, the best choice of 
% fidelity term is s = 2. 
%
%---------------------------------------------- input a grayscale image 
originalImage = imread('coins.png');

%---------------------------------------------- rescale image to [0,1]
originalImage = double(originalImage)./max(max(double(originalImage)));

%---------------------------------------------- remove pixel data 
rng(42)
K = ones(size(originalImage));
g = originalImage;
reductionFactor = 0.75;
for i = 1:size(g,1)
    for j = 1:size(g,2)
        r = rand();
        if r > 1-reductionFactor
            g(i,j) = 0;
            K(i,j) = 0;
        end
    end
end

%---------------------------------------------- set parameters
a = 8E-5;  % Intensity of TV term
b = 1E-4;  % Intensity of curvature term
s = 2;    % Fidelity term in L^s norm

%---------------------------------------------- set options
options = defaultOptions(b,s);
options.showDetails = 1;
options.residualTol = 1E-7;
%---------------------------------------------- run denoising algorithm
u_inpaint = eulerElastica(g,a,b,s,options,K);

%---------------------------------------------- show original image
figure, imagesc(originalImage); colormap(gray); pause(0.01)

close all








%--------------------------------------------------------------------------
%------------------------ Inpainting example  -----------------------------
%--------------------------------------------------------------------------
%
% This example differs from the above in that a parallel version of the
% algorithm is employed. Note that this test problem is really too small
% for parallelism, and that there is no substantial computational time
% saved unless images are of size 400x400 or above.


%---------------------------------------------- set options
options.adaptivity = 1;
options.useParallel = 1;

%---------------------------------------------- run denoising algorithm
u_inpaint = eulerElastica(g,a,b,s,options,K);

%---------------------------------------------- show original image
figure, imagesc(originalImage); colormap(gray); pause(0.01)
