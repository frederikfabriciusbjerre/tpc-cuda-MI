#ifndef CALC_STATISTICS_CUH
#define CALC_STATISTICS_CUH

#include <cmath>
#include <cuda_runtime.h>
#include "pt_old.cuh"  
__device__ const double target_pvals[] = {
    0.06911365, 
    0.05404946, 
    0.1156236, 
    0.1779667, 
    0.2652843, 
    0.1644918, 
    0.1779667, 
    0.2652843, 
    0.122798, 
    0.1644918, 
    0.122798, 
    0.1156236, 
    0.06911365, 
    0.05404946 
};
__device__ bool is_close(double a, double b, double tol) {
    return fabs(a - b) < tol;
}
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
    if (B > 1e-12) {
        double temp = (W / B) * (M / (M + 1.0));
        df = (M - 1) * (1.0 + temp) * (1.0 + temp);
        if (df > 10000) df = INFINITY;
    } else {
        df = INFINITY;
    }

    // return p-value

    double p_val = 2.0 * (1.0 - pt(fabs(ts), df)); 
    // double p_val = 2.0 * pt(fabs(ts), df, 0, 0);
    const double tol = 1e-6;

    // Check if p_val matches any of the target p-values
    for (int i = 0; i < 14; ++i) {
        if (is_close(p_val, target_pvals[i], tol)) {
            printf("p_val: %.8f matches target %.8f\n", p_val, target_pvals[i]);
            printf("df: %f, ts: %f, B: %f, idea: %f \n", df, ts, B, 2.0 * (1.0 - pt(0.0, df)) );
            break; // Exit the loop once a match is found
        }
    }
    return p_val;
}

#endif // CALC_STATISTICS_CUH
