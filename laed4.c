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
    mxDouble *D, *Z; /* pointers to input & output matrices*/
    mxDouble *rho;
    mxDouble *DELTA, *DLAM;
#else
    double *D, *Z; /* pointers to input & output matrices*/
    double *rho;
    double *DELTA, *DLAM;
#endif
    ptrdiff_t n,i;      /* matrix dimensions */
    ptrdiff_t INFO;

#if MX_HAS_INTERLEAVED_COMPLEX
    D = mxGetDoubles(prhs[0]); /* first input matrix */
    Z = mxGetDoubles(prhs[1]); /* second input matrix */
    rho = mxGetDoubles(prhs[2]);
    i = mxGetInt(prhs[3]);
#else
    D = mxGetPr(prhs[0]); /* first input matrix */
    Z = mxGetPr(prhs[1]); /* second input matrix */
    rho = mxGetPr(prhs[2]);
    i = (int)*(double*)mxGetData(prhs[3]);
#endif
    /* dimensions of input matrices */
    n = mxGetM(prhs[0]);

    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    plhs[1] = mxCreateDoubleScalar(mxREAL);
    plhs[2] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);

#if MX_HAS_INTERLEAVED_COMPLEX
    DELTA = mxGetDoubles(plhs[0]);
    DLAM = mxGetDoubles(plhs[1]);
    INFO = mxGetInt(plhs[2]);
#else
    DELTA = mxGetPr(plhs[0]);
    DLAM = mxGetPr(plhs[1]);
    INFO = (int)*(double*)(plhs[2]);
#endif
    
    dlaed4(&n, &i, D, Z, DELTA, rho, DLAM, &INFO);
}
