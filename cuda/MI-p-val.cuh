#ifndef CALC_STATISTICS_CUH
#define CALC_STATISTICS_CUH

#include <cmath>
#include <cuda_runtime.h>
#include "pt_old.cuh"  
// #include "pt.cuh"
__device__ double compute_MI_p_value(const double* z_m, int M, int nrows, int ord) {
    // calculate avgz
    double z_sum = 0.0;
    for (int m = 0; m < M; m++) {
        z_sum += z_m[m];
    }
    double avgz = z_sum / M;
    // calculate within-imputation variance
    double W = 1.0 / (nrows - 3 - ord);

    // calculate between-imputation variance
    double B_sum = 0.0;
    for (int m = 0; m < M; m++) {
        double diff = z_m[m] - avgz;
        B_sum += diff * diff;
    }
    double B = B_sum / (M - 1.0);

    // calculate total variance
    double TV = W + (1.0 + 1.0 / M) * B;

    // calculate test stat
    double ts = avgz / sqrt(TV);

    // calculate degrees of freedom
    double df;
    if (B > 1e-10) {
        double temp = (W / B) * (M / (M + 1.0));
        df = (M - 1) * (1.0 + temp) * (1.0 + temp);
    } else {
        df = INFINITY;
    }

    // return p-value

    double p_val = 2.0 * (1.0 - pt(abs(ts), df)); 
    // double p_val = 2.0 * pt(fabs(ts), df, 0, 0);
    return p_val;
}

#endif // CALC_STATISTICS_CUH
