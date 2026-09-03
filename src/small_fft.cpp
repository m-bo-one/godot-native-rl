#include "small_fft.h"

#include <cmath>
#include <cstring>

using namespace godot;

namespace {

int smallest_factor(int n) {
    for (int p = 2; p * p <= n; p++) {
        if (n % p == 0) {
            return p;
        }
    }
    return n;
}

// One level of a Cooley-Tukey split: p sub-transforms of length n/p, then one combine over
// the master twiddle table. `work` is shared with the recursion below because a child is
// finished with it before its parent's combine begins.
void fft_step(const float *in_re, const float *in_im, int stride,
        float *out_re, float *out_im, int n, int full,
        const float *cosines, const float *sines, float *work_re, float *work_im) {
    if (n == 1) {
        out_re[0] = in_re[0];
        out_im[0] = in_im[0];
        return;
    }
    const int p = smallest_factor(n);
    const int m = n / p;
    for (int j = 0; j < p; j++) {
        fft_step(in_re + (size_t)j * stride, in_im + (size_t)j * stride, stride * p,
                out_re + j * m, out_im + j * m, m, full, cosines, sines, work_re, work_im);
    }
    const int span = full / n;
    for (int k = 0; k < n; k++) {
        float sum_re = 0.0f;
        float sum_im = 0.0f;
        const int r = k % m;
        for (int j = 0; j < p; j++) {
            const int at = j * m + r;
            const int t = (int)(((long long)j * k * span) % full);
            const float c = cosines[t];
            const float s = sines[t];
            sum_re += out_re[at] * c - out_im[at] * s;
            sum_im += out_re[at] * s + out_im[at] * c;
        }
        work_re[k] = sum_re;
        work_im[k] = sum_im;
    }
    memcpy(out_re, work_re, (size_t)n * sizeof(float));
    memcpy(out_im, work_im, (size_t)n * sizeof(float));
}

} // namespace

void SmallFft::prepare(int n) {
    size = n;
    cosines.resize((size_t)n);
    sines.resize((size_t)n);
    for (int t = 0; t < n; t++) {
        const double angle = -2.0 * 3.14159265358979323846 * (double)t / (double)n;
        cosines[(size_t)t] = (float)cos(angle);
        sines[(size_t)t] = (float)sin(angle);
    }
}

// `re` arrives holding the windowed frame and leaves holding the transform. The input is
// real, so `im` is zeroed here rather than asked for: halving the work by packing the frame
// into a half-length complex transform would buy a few milliseconds for a page of algebra.
void SmallFft::run(float *re, float *im, float *work_re, float *work_im) const {
    std::vector<float> in_re((size_t)size);
    memcpy(in_re.data(), re, (size_t)size * sizeof(float));
    std::vector<float> in_im((size_t)size, 0.0f);
    fft_step(in_re.data(), in_im.data(), 1, re, im, size, size,
            cosines.data(), sines.data(), work_re, work_im);
}
