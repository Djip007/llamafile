#include <immintrin.h>

// qq intrinsic manquante
// vectorized :
extern __inline __m512bh
__attribute__ ((__gnu_inline__, __always_inline__, __artificial__))
_mm512_castsi512_bh (__m512i __A)
{
    return (__m512bh) (__A);
}
extern __inline __m256bh
__attribute__ ((__gnu_inline__, __always_inline__, __artificial__))
_mm256_castsi256_bh (__m256i __A)
{
    return (__m256bh) (__A);
}
extern __inline __m512i
__attribute__ ((__gnu_inline__, __always_inline__, __artificial__))
_mm512_castbh_si512 (__m512bh __A)
{
    return (__m512i) (__A);
}

//__m128bh B = _mm256_cvtneps_pbh(_mm256_loadu_ps(pB+j*ldb+k2));
//    __m512i _mm512_broadcastd_epi32 (__m128i a)
//    auto _B = _mm512_broadcastd_2pbh(B);

extern __inline __m512bh
__attribute__ ((__gnu_inline__, __always_inline__))
_mm512_broadcastd_2pbh(__m128bh __A)
{
    return (__m512bh) _mm512_broadcastd_epi32((__m128i)__A);
}

//__m128i _mm_bslli_si128 (__m128i a, int imm8)
extern __inline __m128bh
__attribute__ ((__gnu_inline__, __always_inline__))
_mm_shiftl_2pbh(__m128bh __A)
{
    return (__m128bh) _mm_bsrli_si128((__m128i)__A, 4); // 2xbf16 == 4 char!
}
