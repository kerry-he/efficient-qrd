/*=========================================================
 * matrixMultiply.c - Example for illustrating how to use 
 * BLAS within a C MEX-file. matrixMultiply calls the 
 * BLAS function dgemm.
 *
 * C = matrixMultiply(A,B) computes the product of A*B,
 *     where A, B, and C are matrices containing real values.
 *
 * This is a MEX-file for MATLAB.
 * Copyright 2009-2018 The MathWorks, Inc.
 *=======================================================*/

#include "mex.h"
#include "lapack.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
#if MX_HAS_INTERLEAVED_COMPLEX
    mxDouble *D, *Q, *W, *S; /* pointers to input & output matrices*/
    mxDouble *rho, *DLAMBDA;
#else
    double *D, *W, *Q, *S; /* pointers to input & output matrices*/
    double *rho, *DLAMBDA;
#endif
    ptrdiff_t one = 1;
    ptrdiff_t n, INFO;

#if MX_HAS_INTERLEAVED_COMPLEX
    DLAMBDA = mxGetDoubles(prhs[0]);
    W = mxGetDoubles(prhs[1]);
    rho = mxGetDoubles(prhs[2]);
#else
    DLAMBDA = mxGetPr(prhs[0]);
    W = mxGetPr(prhs[1]);
    rho = mxGetPr(prhs[2]);
#endif
    /* dimensions of input matrices */
    n = mxGetM(prhs[0]);

    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(n, n, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(n, n, mxREAL);
    plhs[3] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);

#if MX_HAS_INTERLEAVED_COMPLEX
    D = mxGetDoubles(plhs[0]);
    Q = mxGetDoubles(plhs[1]);
    S = mxGetDoubles(plhs[2]);
    INFO = mxGetInt(plhs[3]);
#else
    D = mxGetPr(plhs[0]);
    Q = mxGetPr(plhs[1]);
    S = mxGetPr(plhs[2]);
    INFO = (int)*(double*)(plhs[3]);
#endif

    dlaed9(&n, &one, &n, &n, D, Q, &n, rho, DLAMBDA, W, S, &n, &INFO);
}
