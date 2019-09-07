#include <fftw3.h>
#include <complex>


class FastSpectralCollision {
public:
    FastSpectralCollision(int n0, const int& n1);
    ~FastSpectralCollision();
    void fft_test(fftw_complex* f);
private:
    int n;
    fftw_complex* _in;
    fftw_complex* _out;
    fftw_plan p;
};
