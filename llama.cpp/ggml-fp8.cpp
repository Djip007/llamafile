#include <cassert>
#include <cmath>
#include <algorithm>

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml.h"

#include "ggml-fp8.h"

//#define target_clones(x86,arm) __attribute__((__target_clones__(x86 ",default")))
//#define target_clones(x86,arm) __attribute__((__target_clones__(arm ",default")))
//target_clones("arch=znver4,avx512f,avx2","simd")
#ifdef __x86_64__
#define target_clones() __attribute__((__target_clones__("arch=znver4,avx512f,avx2,default"),flatten))
#elif defined(__aarch64__)
// https://gcc.gnu.org/onlinedocs/gcc/AArch64-Options.html
// result in: undefined reference to `__aarch64_cpu_features'
//#define target_clones() __attribute__((__target_clones__("simd,default"),flatten))
#define target_clones() __attribute__((flatten))
#else
#define target_clones()
#endif

/*
> build:
make clean
make -j16
make -j16 install PREFIX=/home/philou/LLM/usr/

./usr/bin/llamafile-quantize Mistral-Nemo-Instruct-2407.BF16.gguf Mistral-Nemo-Instruct-2407.E3M4_Q.gguf E3M4_Q

./usr/bin/llamafile -m Mistral-Nemo-Instruct-2407.E3M4_Q.gguf -c 128 -n 16 -t 0 -s 42 -p "[INST]bonjour a tu un nom. je ne sais pas comment t'appeler. Si tu n'en as pas je peux t'appeler TINTIN[/INST]"

*/

template<int N> constexpr float EXP2() {
    if constexpr (N==0) return 1;
    if constexpr (N>0) return EXP2<N-1>()*2;
    if constexpr (N<0) return EXP2<N+1>()/2;
}

// 2^N avec N>0 en entier
template<int N> constexpr int EXP_I2() {
    if constexpr (N==0) return 1;
    if constexpr (N>0) return EXP_I2<N-1>()*2;
}

template<int _E> //, int M=7-E>  1.7 bits!
struct FP8 {
    uint8_t bits;
    using type = FP8<_E>;
    static constexpr int E=_E;
    static constexpr int M=7-_E;
    static constexpr int E_BIAS=EXP2<_E-1>()-1;
    static constexpr float MAX() { return (2-EXP2<-M+1>())*EXP2<EXP_I2<_E-1>()>(); }
    static constexpr float MIN() { return EXP2<-M>()*EXP2<2-EXP_I2<_E-1>()>(); }
    //=============================================

    #pragma omp declare simd
    void operator=(float value) {
        union {
            float f;
            uint32_t bits;
        } in = {value};
        // le signe:
        bits = (in.bits >> 24) & 0x80;
        // la valeur sans la signe!
        in.bits &= 0x7fffffff;
        //GGML_ASSERT(in.bits < 0x7f800000); // +/- infini ou NAN
        if (in.f >= MAX()) {
            bits |= 0x7E;
        } else if (in.f<MIN()) { // => 0.
            // OK: S.0000000
        } else {
            in.f *= EXP2<E_BIAS-127>();
            in.bits += 1<<(22-M); // for rounding
            bits |= (in.bits >> (23-M)) & 0x7F;
        }
    }

    #pragma omp declare simd
    operator float () const {
        union {
            float f;
            uint32_t bits;
        } out = {0};
        // le signe:
        out.bits = bits & 0x80;
        out.bits <<= 24;
        uint32_t _bits = bits & 0x7F;
        _bits <<= (23-M);
        out.bits |= _bits;
        out.f *= EXP2<127-E_BIAS>();
        return out.f;
    }
};

template<int E>
target_clones()
static inline void conv(const FP8<E>* x, float* y, int64_t size) {
    #pragma omp simd
    for (int64_t i=0; i<size; i++) {
        y[i] = (float) x[i];
    }
}

// [[ attribute ]]
template<int E>
target_clones()
static inline void conv(const float* x, FP8<E>* y, int64_t size) {
    #pragma omp simd
    for (int64_t i=0; i<size; i++) {
        y[i] = x[i];
    }
}

template<int E>
target_clones()
static inline float dot(const FP8<E>* x, const float* y, int64_t size) {
    float z = 0;
    #pragma omp simd reduction(+:z)
    for (int64_t i=0; i<size; i++) {
        z += ((float)x[i])*y[i];
    }
    return z;
}

template <int E, int QK>
struct bloc_fp8 {
    float d;
    FP8<E> qs[QK];
};

template <int E, int QK>
target_clones()
static inline void conv(const bloc_fp8<E, QK>* x, float* y, int64_t size) {
    const auto qk_size = size / QK;
    for (int64_t q=0; q<qk_size; ++q) {
        #pragma omp simd
        for (int64_t i=0; i<QK; i++) {
            y[q*QK+i] = ((float) x[q].qs[i])*(x[q]).d;
        }
    }
}

template <int E, int QK>
target_clones()
static inline void conv(const float* x, bloc_fp8<E, QK>* y, int64_t size) {
    const auto qk_size = size / QK;
    for (int64_t q=0; q<qk_size; ++q) {
        float m = 0;
        #pragma omp simd reduction(max:m)
        for (int64_t i=0; i<QK; i++) {
            m = std::max(std::abs(x[q*QK+i]),m);
        }
        const float D = FP8<E>::MAX()/m;
        y[q].d = m/FP8<E>::MAX();
        #pragma omp simd
        for (int64_t i=0; i<QK; i++) {
            y[q].qs[i] = x[q*QK+i]*D;
        }
    }
}

template <int E, int QK>
target_clones()
static inline float dot(const bloc_fp8<E, QK>* x, const float* y, int64_t size) {
    float z = 0;
    const auto qk_size = size / QK;
    for (int64_t q=0; q<qk_size; ++q) {
        float z0 = 0;
        #pragma omp simd reduction(+:z0)
        for (int64_t i=0; i<QK; i++) {
            z0 += ((float)x[q].qs[i])*y[q*QK+i];
        }
        z += (x[q]).d * z0;
    }
    return z;
}

// the C API.
void ggml_e5m2_to_fp32_row(const ggml_e5m2_t * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    conv(reinterpret_cast<const FP8<5>*>(x), y, k);
}
void ggml_fp32_to_e5m2_row(const float * GGML_RESTRICT x, ggml_e5m2_t * GGML_RESTRICT y, int64_t k) {
    conv(x, reinterpret_cast<FP8<5>*>(y), k);
}
void ggml_fp32_to_e5m2_row_ref(const float * GGML_RESTRICT x, ggml_e5m2_t * GGML_RESTRICT y, int64_t k) {
    for (int64_t i =0; i<k; ++i) {
        reinterpret_cast<FP8<5>*>(y)[i] = x[i];
    }
}

void ggml_e4m3_to_fp32_row(const ggml_e4m3_t * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    conv(reinterpret_cast<const FP8<4>*>(x), y, k);
}
void ggml_fp32_to_e4m3_row(const float * GGML_RESTRICT x, ggml_e4m3_t * GGML_RESTRICT y, int64_t k) {
    conv(x, reinterpret_cast<FP8<4>*>(y), k);
}
void ggml_fp32_to_e4m3_row_ref(const float * GGML_RESTRICT x, ggml_e4m3_t * GGML_RESTRICT y, int64_t k) {
    for (int64_t i =0; i<k; ++i) {
        reinterpret_cast<FP8<4>*>(y)[i] = x[i];
    }
}

void dequantize_row_e4m3_q(const block_e4m3_q * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    conv(reinterpret_cast<const bloc_fp8<4, QK_K>*>(x), y, k);
}
void quantize_row_e4m3_q(const float * GGML_RESTRICT x, block_e4m3_q * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    conv(x, reinterpret_cast<bloc_fp8<4, QK_K>*>(y), k);
}
void quantize_row_e4m3_q_ref(const float * GGML_RESTRICT x, block_e4m3_q * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    conv(x, reinterpret_cast<bloc_fp8<4, QK_K>*>(y), k);
}

void dequantize_row_e3m4_q(const block_e3m4_q * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    conv(reinterpret_cast<const bloc_fp8<3, QK_K>*>(x), y, k);
}
void quantize_row_e3m4_q(const float * GGML_RESTRICT x, block_e3m4_q * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    conv(x, reinterpret_cast<bloc_fp8<3, QK_K>*>(y), k);
}
void quantize_row_e3m4_q_ref(const float * GGML_RESTRICT x, block_e3m4_q * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    conv(x, reinterpret_cast<bloc_fp8<3, QK_K>*>(y), k);
}

// the dot product for FP8 weight
void ggml_vec_dot_e5m2(int n, float * GGML_RESTRICT s, size_t bs, const ggml_e5m2_t * GGML_RESTRICT vx, size_t bx, const float * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);
    *s = dot(reinterpret_cast<const FP8<5>*>(vx), vy, n);
}

void ggml_vec_dot_e4m3(int n, float * GGML_RESTRICT s, size_t bs, const ggml_e4m3_t * GGML_RESTRICT vx, size_t bx, const float * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);
    *s = dot(reinterpret_cast<const FP8<4>*>(vx), vy, n);
}

void ggml_vec_dot_e4m3_q(int n, float * GGML_RESTRICT s, size_t bs, const block_e4m3_q * GGML_RESTRICT vx, size_t bx, const float * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);
    *s = dot(reinterpret_cast<const bloc_fp8<4, QK_K>*>(vx), vy, n);
}

void ggml_vec_dot_e3m4_q(int n, float * GGML_RESTRICT s, size_t bs, const block_e3m4_q * GGML_RESTRICT vx, size_t bx, const float * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);
    *s = dot(reinterpret_cast<const bloc_fp8<3, QK_K>*>(vx), vy, n);
}
