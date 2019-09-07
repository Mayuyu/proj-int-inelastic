#include <fftw3.h>
#include <iostream>
#include <complex>
#include "fft_test.h"

FastSpectralCollision::FastSpectralCollision(const int& N) {
    n0 = N;
    _in = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N);
    _out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N);
    p = fftw_plan_dft_2d(int n0, int n1, fftw_complex *_in, fftw_complex *out,)
}

void FastSectralCollision::fft_test(fftw_complex *f) {

}
