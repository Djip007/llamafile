// gestion des types pour les templates.
#include "ggml.h"

// voir quel type gerer...
// - quel format
// - quel possibilié:
//     - nan          => ici jamais utils
//     - subnormal    => ?
//     - exp/bias  :   On pourais voir cas ou MAX==1
//  Combinaison a Traiter:
//   - cas reel complet: EXMX
//   - MAX=1 / pas NAN /
//      => sub/normal:   EXMX[s/f]
typedef struct { uint8_t bits; } ggml_E5M2_t;
typedef struct { uint8_t bits; } ggml_E4M3_t;
typedef struct { uint8_t bits; } ggml_E3M4_t;

using fp32_t = float;
using bf16_t = ggml_bf16_t;
using fp16_t = ggml_fp16_t;
//using fp8_t  = ggml_fp8_t;    // == f8_E3M4_t ??
using f8_E3M4_t = ggml_E3M4_t;  // GGML_TYPE_FP8 => GGML_TYPE_FP8_E3M4
using f8_E4M3_t = ggml_E4M3_t;  // GGML_TYPE_FP8 => GGML_TYPE_FP8_E4M3
using f8_E5M2_t = ggml_E5M2_t;  // GGML_TYPE_FP8 => GGML_TYPE_FP8_E5M2


// qq veteur de bf16
using bf16_x32 = __m512bh;
using bf16_x16 = __m256bh;
using bf16_x8  = __m128bh;

/*
FP8_NORMAL
fp8<E,M>: S.[E].[M]

*/
template<int N> float EXP2() {
    if constexpr (N==0) return 1;
    if constexpr (N>0) return EXP2<N-1>()*2;
    if constexpr (N<0) return EXP2<N+1>()/2;
}

template<int E> //, int M=7-E>  1.7 bits!
struct FP8 {
    uint8_t bits;
    using type = FP8<E>;
    static constexpr int M=7-E;
    static constexpr int E_BIAS=EXP2<E>()-1;
    // MAX = 1+1/2+1/4...1/(2^(M+1))
    static constexpr float MAX() {
        if constexpr (M==0) {
            return 1;
        }
        return FP8<E+1>::MAX()+EXP2<M+1>();
    }
    // MIN = (1/2)^(M)
    static constexpr float MIN() {
        return EXP2<-M>();
    }

    void operator=(fp32_t value) {
        // GGML_ASSERT(!isnan(value));
        union {
            float f;
            uint32_t bits;
        } in = {value};
        // le signe:
        bits = (in.bits >> 24) & 0x80;
        // la valeur sans la signe!
        in.bits &= 0x7fffffff;
        GGML_ASSERT(in.bits < 0x7f800000); // +/- infini ou NAN
        if (in.f >= MAX()) {
            bits |= 0x7F; // 0x7E si NAN...
        } else if (in.f<MIN()) { // => 0.
            // On est bon: que le signes S.0000000
        } else {
            int exp = (in.bits>>23 & 0xFF)-127;
            uint8_t exp_bits = (exp + E_BIAS) & 0xF;
            // TODO:
        }
        //
    }
    // ca serais mieux avec ca:
    //void operator=(bf16_t value) {
    //    ;
    //}
    static bf16_x32 load(type* ptr);
private:

};

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
struct _f8_34_t {
    union {
        ggml_E3M4_t f;
        uint8_t b=0x80; // +0?
    };
};
struct _f8_43_t {
    union {
        ggml_E4M3_t f;
        uint8_t b=0x80; // +0?
    };
};
struct _f8_52_t {
    union {
        ggml_E5M2_t f;
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

//using bf16_2x16_t = _bf16_2x16_t;
//using fp32_16x1_t = _fp32_16x1_t;

// les usages:
//============================================
//MAX<T>()  // pour les cas qui ont besoin d'un scale!
template<typename T> float MAX() {static_assert(false, "doit etre surchargé pour ce type"); return 0;}
// les conversion a implementer: T2 => T1
//> conv(T1 dest, const T2 orig): dest = orig
template<typename T1, typename T2> static inline void conv(T1& dest , const T2& orig) {dest = orig;}
//> bf16x32_t load(T orig): T => bf16_x32
//

template<> float MAX<f8_E4M3_t>() {return 448;}
// [...]
//============================================
// - cas general: pour B:
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
// fp32_t => bf16x32_t
static inline auto load(const fp32_t *X) {
    auto x1 = _mm512_loadu_ps(X);
    auto x2 = _mm512_loadu_ps(X+16);
    return _mm512_cvtne2ps_pbh(x2,x1);
}

//============================================
// cas pour A:
//------------------------------------------
// => fp16
static inline auto load(const bf16_t *X) {
    return (__m512bh) _mm512_loadu_epi16(X);
}

//------------------------------------------
// => f8_E4M3:
// avec staturation !! (devrais pouvoir etre supprimé si on code le suivant sans lui...)
static inline void conv(f8_E4M3_t& dest, const fp32_t orig) {
    _fp32_t u;
    u.f = orig;
    // le signe:
    dest.bits = (u.b >> 24) & 0x80;
    // la valeur sans la signe!
    u.b &= 0x7fffffff;
    if (u.b > 0x7f800000) { // +/-nan
        GGML_ASSERT(u.b <= 0x7f800000);
        dest.bits |= 0x7F;
    } else if (u.f>=448) { // +/-max : saturation!
        dest.bits |= 0x7E;
    } else if (u.f<0.001953125) { // +/-0 : trop petit.
        // dest.bits &= 0x80;
    } else if (u.b == 0x7f800000) {  // +/- infinite : saturation
        // est-ce que ca arrive ou deja traité par le precedant if?
        GGML_ASSERT(u.b <= 0x7f800000);
        dest.bits |= 0x7F;
    } else {
        //int exp = floorf(log2f(f));
        int exp = (u.b>>23 & 0xFF)-127;
        uint8_t exp_bits = (exp + 7) & 0xF;
        if (exp < -6) { // subnormal
            exp_bits = 0;
            auto mantissa = u.f / exp2f(-6);
            uint8_t mantissa_bits = (uint8_t)(mantissa * 8) & 0x7;
            dest.bits |= mantissa_bits;
        } else {
            // la mantise est bonne y a que à la copier:
            dest.bits |= (u.b>>20) & 0x7;
        }
        // if (exp_bits == 15 && mantissa_bits == 0x07) mantissa_bits = 6; // si pas de nan on peu supprimer...
        dest.bits |= (exp_bits << 3); // | mantissa_bits;
    }
}
// bf16 => f8_E4M3 :
static inline void conv(f8_E4M3_t& dest, const bf16_t orig) {
    fp32_t f;
    conv(f,orig);
    conv(dest,f);
}
/*
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
*/
static inline __m512bh
llamafile_fp8_e4m3_to_bf16_avx512(__m256i fp8_vec)
{
    __m512i x = _mm512_cvtepu8_epi16(fp8_vec);
    __m512i lg2mant = _mm512_mask_mov_epi16(
      _mm512_mask_mov_epi16(_mm512_setzero_si512(),
                            _mm512_test_epi16_mask(x, _mm512_set1_epi16(2)),
                            _mm512_set1_epi16(1)),
      _mm512_test_epi16_mask(x, _mm512_set1_epi16(4)),
      _mm512_set1_epi16(2));
    return _mm512_castsi512_bh(_mm512_or_si512(
      _mm512_maskz_mov_epi16(
        _mm512_cmpneq_epi16_mask(_mm512_and_si512(x, _mm512_set1_epi16(127)),
                                 _mm512_setzero_si512()),
        _mm512_mask_blend_epi16(
          _mm512_test_epi16_mask(x, _mm512_set1_epi16(120)),
          _mm512_or_si512(
            _mm512_and_si512(_mm512_sllv_epi16(
                               _mm512_and_si512(x, _mm512_set1_epi16(3)),
                               _mm512_sub_epi16(_mm512_set1_epi16(7), lg2mant)),
                             _mm512_set1_epi16(0x007f)),
            _mm512_slli_epi16(_mm512_add_epi16(lg2mant, _mm512_set1_epi16(118)),
                              7)),
          _mm512_or_si512(
            _mm512_slli_epi16(_mm512_and_si512(x, _mm512_set1_epi16(7)), 4),
            _mm512_slli_epi16(
              _mm512_add_epi16(
                _mm512_srli_epi16(_mm512_and_si512(x, _mm512_set1_epi16(120)),
                                  3),
                _mm512_set1_epi16(120)),
              7)))),
      _mm512_slli_epi16(_mm512_and_si512(x, _mm512_set1_epi16(128)), 8)));
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
// TODO: separer les cas avec et sans saturation.. et sans NAN?
// f8_E4M3s_t f8_E4M3n_t
static inline auto load(const f8_E4M3_t *X) {
    auto x = _mm256_loadu_epi8(X);
    return llamafile_fp8_e4m3_to_bf16_avx512_nd(x);
    //return llamafile_fp8_e4m3_to_bf16_avx512(x);
}

//------------------------------------------------------------------------------------
//  pour C et B_cache.
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
