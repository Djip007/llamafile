// gestion des types pour les templates.
#include "ggml.h"

typedef struct { uint8_t bits; } ggml_E5M2_t;
typedef struct { uint8_t bits; } ggml_E4M3_t;
typedef struct { uint8_t bits; } ggml_E3M4_t;

struct _fp32_t {
    union {
        float f;
        uint32_t b=0x80000000;
    }; // x;
};
struct _bf16_t {
    union {
        ggml_bf16_t f;
        uint16_t b=0x8000; // +0?
    };
};
struct _f8_t {
    union {
        ggml_E4M3_t f;
        uint8_t b=0x80; // +0?
    };
};
// qq types composés
struct _bf16_2x16_t {
    union {
        __m512bh f; // acce en bf16 @ privilegier
        __m512i  e; // acce "entier" pour certain intrinsec.
    };
};
struct _fp32_16x1_t {
    union {
        __m512  f; // acce en fp32 @ privilegier
        __m512i e; // acce "entier" pour certain intrinsec.
    };
};

using fp32_t = float;
using bf16_t = ggml_bf16_t;
using fp16_t = ggml_fp16_t;
//using fp8_t  = ggml_fp8_t;  // GGML_TYPE_FP8 => GGML_TYPE_FP8_E4M3
//using f8_E4M3_t = ggml_fp8_t;  // GGML_TYPE_FP8 => GGML_TYPE_FP8_E4M3
//using bf16_2x16_t = _bf16_2x16_t;
//using fp32_16x1_t = _fp32_16x1_t;

// les usages:
//============================================
//> conv(T1 dest, T2 orig): dest = orig
// - cas general
// T2 => T1
template<typename T1, typename T2> static inline void conv(T1& dest , const T2& orig) {dest = orig;}
// fp32 => bf16:
static inline void conv(bf16_t& dest, const fp32_t orig) {
    //dest = ggml_compute_fp32_to_bf16(orig);
    _fp32_t u;
    u.f = orig;
    if ((u.b & 0x7fffffff) > 0x7f800000) { /* nan */
        dest.bits = (u.b >> 16) | 64; /* force to quiet */
    } else {
        dest.bits = (u.b + (0x7fff + ((u.b >> 16) & 1))) >> 16;
    }
}
// bf16 => fp32 :
static inline void conv(fp32_t& dest, const bf16_t orig) {
    //dest = ggml_compute_bf16_to_fp32(orig);
    _fp32_t u;
    u.b = (uint32_t)orig.bits << 16;
    dest = u.f;
}

//------------------------------------------
//> bf16x32_t load(T orig): T => bf16_x32
static inline auto load(const fp32_t *X) {
    auto x1 = _mm512_loadu_ps(X);
    auto x2 = _mm512_loadu_ps(X+16);
    return _mm512_cvtne2ps_pbh(x2,x1);
}
static inline auto load(const bf16_t *X) {
    return (__m512bh) _mm512_loadu_epi16(X);
}

//------------------------------------------
//> store(T* , T[N])
static inline void store(bf16_t *pX, const __m512bh& x) {
    _mm512_storeu_epi16(pX, (__m512i)x);
}
static inline void store(fp32_t *pX, const __m512& x) {
    _mm512_storeu_ps(pX, x);
}
/*
// write C after last reduction
template<typename... T>
static inline void store(fp32_t *pX, T&&... x) {
    constexpr __mmask16 _m = ((1<<sizeof...(T))-1);
    auto pack = hadd(std::forward<T>(x)...);
    _mm512_mask_storeu_ps(pX, _m, pack);
}
*/

//> fp32_t hsum(fp32x16_t) : somme les 16 float du vecteur
// y appliquer un facteur de correction si besoin
template<bool SCALE=false>
static inline float hsum(__m512 x, fp32_t scale=1) {
    if constexpr (SCALE) {
        return scale*_mm512_reduce_add_ps(x);
    } else {
        return _mm512_reduce_add_ps(x);
    }
}

//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------
// => implemetation des OPs

//------------------------------------------------------------------------------------
// - fp8
/*
static inline f8_E4M3_t
llamafile_fp32_to_fp8_e4m3(fp32_t f) {
    union {
        unsigned char i;
        f8_E4M3_t f;
    } out{0};
    uint8_t sign = signbit(f) ? 128 : 0;
    if (isnan(f)) {
        out.i = sign | 127;
    } else if (!f) {
        out.i = sign;
    } else {
        f = fabsf(f);
        int exp = floorf(log2f(f));
        float mantissa = f / exp2f(exp) - 1;
        if (exp < -6) {
            mantissa = f / exp2f(-6); // subnormal
            exp = -7;
        }
        if (exp > 8) {
            out.i = sign | 0x7E; // overflow
        } else {
            uint8_t exp_bits = (exp + 7) & 15;
            uint8_t mantissa_bits = (uint8_t)(mantissa * 8) & 7;
            // [jpp] avoid generate NAN ?
            if (exp_bits == 15 && mantissa_bits == 0x07)
                mantissa_bits = 6;
            out.i = sign | (exp_bits << 3) | mantissa_bits;
        }
    }
    return out.f;
}

// TODO: reprande avec le cas suivant & celui de [jart] comment faire plus vite!
static inline __m512bh
llamafile_fp8_e4m3_to_bf16_avx512(__m256i fp8_vec){
    // extract components:
    __m256i expo_8 = _mm256_and_si256(fp8_vec, _mm256_set1_epi8(0x78));
    __m256i mant_8 = _mm256_and_si256(fp8_vec, _mm256_set1_epi8(0x07));
    __m256i sign_8 = _mm256_and_si256(fp8_vec, _mm256_set1_epi8(0x80));

    // denorm mask
    //> need AVX512BW + AVX512VL ?
    __mmask32 is_denorm = _mm256_cmpeq_epi8_mask(expo_8, _mm256_setzero_si256());
    __m512i expo_16 = _mm512_cvtepu8_epi16(expo_8);
    __m512i mant_16 = _mm512_cvtepu8_epi16(mant_8);
    __m512i sign_16 = _mm512_cvtepu8_epi16(sign_8);
    //> pure AVX512F:
    //__mmask32 is_denorm = _mm512_cmpeq_epi16_mask(expo_16, _mm512_setzero_si512());
    __mmask16 is_denorm_low  = is_denorm;
    __mmask16 is_denorm_high = is_denorm>>16;

    // shift
    expo_16 = _mm512_slli_epi16(_mm512_add_epi32(expo_16,_mm512_set1_epi16(120<<3)), 4);
    mant_16 = _mm512_slli_epi16(mant_16, 4);
    sign_16 = _mm512_slli_epi16(sign_16, 8);

    // correction denorm exp:
    expo_16 = _mm512_mask_blend_epi16(is_denorm, expo_16, _mm512_set1_epi16((-6 + 127) << 7));

    __m512i em = _mm512_or_si512(expo_16,mant_16);

    // correction denorm mantissa using fp32 Aritmetics:
    __m256bh low_bh  = _mm256_castsi256_bh(_mm512_castsi512_si256(em));
    __m256bh high_bh = _mm256_castsi256_bh(_mm512_extracti32x8_epi32 (em, 1));
    __m512 low  = _mm512_cvtpbh_ps( low_bh);
    __m512 high = _mm512_cvtpbh_ps(high_bh);
    low  = _mm512_mask_add_ps( low, is_denorm_low ,  low, _mm512_set1_ps(-1.0/64));
    high = _mm512_mask_add_ps(high, is_denorm_high, high, _mm512_set1_ps(-1.0/64));
    __m512bh result = _mm512_cvtne2ps_pbh(high,low);

    return _mm512_castsi512_bh(_mm512_or_si512(sign_16,_mm512_castbh_si512(result)));
}

static inline __m512bh
llamafile_fp8_e4m3_to_bf16_avx512_nd(__m256i fp8_vec) {
    __m512i fp8_v16 = _mm512_cvtepu8_epi16(fp8_vec);

    // denorm mask  => need AVX512BW ?
    __mmask32 is_denorm = _mm512_testn_epi16_mask(fp8_v16, _mm512_set1_epi16(0x78));

    __m512i mant_16 = _mm512_and_si512(fp8_v16, _mm512_set1_epi16(0x7F));
    __m512i sign_16 = _mm512_and_si512(fp8_v16, _mm512_set1_epi16(0x80));

    // shift
    mant_16 = _mm512_slli_epi16(_mm512_add_epi32(mant_16,_mm512_set1_epi16(120<<3)), 4);
    sign_16 = _mm512_slli_epi16(sign_16, 8);

    __m512i em = _mm512_mask_blend_epi16(is_denorm,mant_16,_mm512_setzero_si512());
    return _mm512_castsi512_bh(_mm512_or_si512(sign_16,em));
}

static inline auto load(const f8_E4M3_t *X) {
    auto x = _mm256_loadu_epi8(X);
    return llamafile_fp8_e4m3_to_bf16_avx512_nd(x);
    // return llamafile_fp8_e4m3_to_bf16_avx512(x);
}
*/

