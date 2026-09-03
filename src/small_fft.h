#ifndef SMALL_FFT_H
#define SMALL_FFT_H

#include <vector>

namespace godot {

// A complex FFT for a length whose factors are small. It splits on the smallest factor and
// recurses, which for the 400-point window of one front end and the 320-point window of
// another is a few thousand operations per frame against a hundred thousand for a direct sum.
struct SmallFft {
    int size = 0;
    std::vector<float> cosines;
    std::vector<float> sines;

    void prepare(int n);
    // Transforms one real frame in place into `re`/`im`, using `work` as scratch. All four
    // arrays are `size` long; `work` may be shared between calls on the same thread.
    void run(float *re, float *im, float *work_re, float *work_im) const;
};

} // namespace godot

#endif // SMALL_FFT_H
