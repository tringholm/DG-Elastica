function [u, energy] = eulerElastica(g,K,a,b,s,options)

if nargin == 5
    disp('No options specified, using default options.')
    options = defaultOptions(b,s);
elseif nargin ~= 6
    disp('Too few or too many input arguments.')
    u = g;
    energy = inf;
    return
end

if options.useParallel
    if exist('dgstepMexPara','file') ~= 3
        mex dgstepMexPara.c
    end
    if exist('gradCurvMexPara','file') ~= 3
        mex gradCurvMexPara.c
    end
    if exist('partitionMex','file') ~= 3
        mex partitionMex.c
    end
    [u, energy] = eulerElasticaMexPara(g,K,a,b,s,options);
else
    if options.useMex
        if exist('dgstepMex','file') ~= 3
            mex dgstepMex.c
        end
        if exist('gradCurvMex','file') ~= 3
            mex gradCurvMex.c
        end
        [u,energy] = eulerElasticaMex(g,K,a,b,s,options);
    else
        [u,energy] = eulerElasticaMatlab(g,K,a,b,s,options);
    end
end

end