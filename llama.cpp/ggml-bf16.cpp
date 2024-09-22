/*
#define __x86_64__
#define __SIZE_TYPE__    unsigned int
namespace std {
    typedef __SIZE_TYPE__ size_t;
}
typedef __SIZE_TYPE__ size_t;
*/
/*
> build:
make clean
make -j16
make -j16 install PREFIX=/home/philou/LLM/usr/

// trace des tenseurs
GGML_SCHED_DEBUG=1 GGML_USE_BACKEND_BF16=1 ./usr/bin/llamafile -m Mistral-Nemo-Instruct-2407.BF16.gguf -c 128 -n 2 -t 0 -p "[INST]bonjour a tu un nom. je ne sais pas comment t'appeler. Si tu n'en as pas je peux t'appeler TINTIN[/INST]"

TRACE> MUL_MAT@result_output (s0(output.weight)[5120:131072:1:1/2:10240:1342177280:1342177280]@bf16, s1(output.weight)[5120:128:1:1/4:20480:2621440:2621440]@f32 => [131072:128:1:1/4:524288:67108864:67108864]@f32)
TRACE> MUL_MAT@result_output (s0(output.weight)[5120:131072:1:1/2:10240:1342177280:1342177280]@bf16, s1(output.weight)[5120:1:1:1/4:20480:20480:20480]@f32 => [131072:1:1:1/4:524288:524288:524288]@f32)
TRACE> MUL_MAT@result_output (s0(output.weight)[5120:131072:1:1/2:10240:1342177280:1342177280]@bf16, s1(output.weight)[5120:1:1:1/4:20480:20480:20480]@f32 => [131072:1:1:1/4:524288:524288:524288]@f32)

> run: une fois implementé
GGML_USE_BACKEND_BF16=1 OMP_NUM_THREADS=8 ./usr/bin/llamafile -m Mistral-Nemo-Instruct-2407.BF16.gguf -c 128 -n 16 -t 0 -p "[INST]bonjour a tu un nom. je ne sais pas comment t'appeler. Si tu n'en as pas je peux t'appeler TINTIN[/INST]"
GGML_USE_BACKEND_BF16=1 OMP_NUM_THREADS=8 ./usr/bin/llamafile-bench -m Mistral-Nemo-Instruct-2407.BF16.gguf -n 16 -p "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128,256,512" -r 3

# GGML_USE_BACKEND_BF16=1 OMP_NUM_THREADS=8 GOMP_CPU_AFFINITY="0,2,4,6,8,10,12,14" ./usr/bin/llamafile -m Mistral-Nemo-Instruct-2407.BF16.gguf -c 128 -n 16 -t 0 -p "[INST]bonjour a tu un nom. je ne sais pas comment t'appeler. Si tu n'en as pas je peux t'appeler TINTIN[/INST]"
# GGML_USE_BACKEND_BF16=1 OMP_NUM_THREADS=8 GOMP_CPU_AFFINITY="0,2,4,6,8,10,12,14" ./usr/bin/llamafile-bench -m Mistral-Nemo-Instruct-2407.BF16.gguf -n 16 -p "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128,256,512" -r 3
# celui de reference (llamafile)
./usr/bin/llamafile-bench -m Mistral-Nemo-Instruct-2407.BF16.gguf -n 16 -p "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128,256,512" -r 3

 */

// # Trace/debug: voir GGML_SCHED_DEBUG=1
//     LLAMA_LOG_INFO("%s: model configured\n", __func__);

/*
TODO:
 - revoir la facon de configurer les threads OpenMP...
 -

 */

#include "ggml-bf16.h"

#ifdef __x86_64__
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "json.h" // https://github.com/nlohmann/json?tab=readme-ov-file#specializing-enum-conversion

#include "ggml-x86_64-immintrin.h"

#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <cosmo.h>
#include <string>
#include <regex>
#include <list>

#include "ggml-bf16-log.inc"

//------------------------------------------------------------------------------------
// Tools
// - se deplacer dans un buffer en nombre de byte
static inline void* add_byte(void* addr, std::size_t nb) {
    return (void*) (((char*)addr)+nb);
}

// gestion des types pour les templates.
struct _bf16_t {
    uint16_t u=0x8000; // +0?
};
struct _f8_t {
    uint8_t u; // =0x80; // +0?
};

using fp32_t = float;
using bf16_t = struct _bf16_t;
using f8_E4M3_t = struct _f8_t;  // GGML_TYPE_FP8 => GGML_TYPE_FP8_E4M3

// qq types composés
struct bf16_2x16_t {
    union {
        __m512bh f; // acce en bf16 @ privilegier
        __m512i  e; // acce "entier" pour certain intrinsec.
    };
};
struct fp32_16x1_t {
    union {
        __m512  f; // acce en fp32 @ privilegier
        __m512i e; // acce "entier" pour certain intrinsec.
    };
};
// voir a ajouter qq operateurs ...

template<typename T> inline bool is(const struct ggml_tensor * t) {return false;}
template<> inline bool is<   fp32_t>(const struct ggml_tensor * t) {return t->type==GGML_TYPE_F32;}
template<> inline bool is<   bf16_t>(const struct ggml_tensor * t) {return t->type==GGML_TYPE_BF16;}
template<> inline bool is<f8_E4M3_t>(const struct ggml_tensor * t) {return t->type==GGML_TYPE_FP8;}

namespace ggml::backend::bf16 {
    enum class TYPE {
        // Tags
        NON_SUPPORTE,
        // les types de base sans block (format d'origine)
        FP32,
        FP16,
        BF16,
        FP8, // F8_E4M3
        //E3M4,
        //E3M4_dec,
        // les types stocké sous forme de bloc 2D:
        BF16_32x1,  // tous les K se suivent
        BF16_8x8,   // des block de [8,8]
        BF16_2x16,  // des block de [2,16]  (1 pour dot2!)
        //E3M4_32x1,  // @ voir
        E4M3_32x1,
        E4M3_8x8,
        E4M3_2x16,
        //E5M2_32x1,  // @ voir
        INVALID=-1,
    };

    // TODO: definir le "json" pour cet enum... et p'etre une macro pour le recuperer?
    // + voir si on ne cre pas une map?vector "correcte" avec!
    // et gerer ca avec des regex: https://en.cppreference.com/w/cpp/regex/regex_match
    // "prefix\\.[:digit:]+\\.suffix\\.weight"

    // pour l'instant on ne les autorisent pas tous...
    NLOHMANN_JSON_SERIALIZE_ENUM( TYPE, {
            {TYPE::INVALID, nullptr},
            {TYPE::FP32, "FP32"},
            {TYPE::BF16, "BF16"},
            {TYPE::FP8, "FP8"},
    })
#define DECODE_TYPE(val) val.template get<ggml::bf16::TYPE>()

    class tensor {
    public:
        const TYPE type;
        // float scale = 1; // possible que pour les poids... est-ce utils
        // TODO les tailles des blocs!  M0/N0/M1/N1 ou K0/M0/K1/M1
        //      "constant" => WEIGHT => changement de type et repack possible!
        tensor(TYPE t) : type(t) {}
        virtual ~tensor(){};

        // les enregistrement:
        static void SET(struct ggml_tensor *op, tensor* backend) {
            op->bf16_tensor = backend;
        }
        static tensor* GET(const struct ggml_tensor *op) {
            return (tensor*) op->bf16_tensor;
        }
        // TODO:
        // les methodes a implementer.
        //  is_allowed
        //  size
        //  set
        //

        // - voir les methodes d'accé au données.
        // voir quel methodes pour gerer les "possibles"
        // - le "choix" de type est static
        // - les tenseur dynamique sont static
        // - les tenseur constant sont instancé/calculé

    };

    // TODO: gere TYPE + TAILLE_BLOC!
    static tensor tensor_non_supporte_t(TYPE::NON_SUPPORTE);
    // static tensor tensor_a_convertir(TYPE::A_CONVERTIR);  // ??? pas forcement utils?
    static tensor tensor_fp32_t(TYPE::FP32);
    static tensor tensor_bf16_t(TYPE::BF16);

    static tensor tensor_bf16_32x1_t(TYPE::BF16_32x1);
    static tensor tensor_bf16_2x16_t(TYPE::BF16_2x16);

    // que des instances "static"
    // Arbre de decision ?
    class op {
    public:
        virtual ~op() {}
        // - is_alowed => TODO...
        virtual bool C_is_allowed(const struct ggml_tensor *C) {
            // TODO @ mettre dans un fils!
            // TODO tester avec le type_bf16 de ce tenseur maintenant qu'il est mis
            if (C->type != GGML_TYPE_F32) return false;
            if (ggml_is_transposed(C)) return false;
            return true;
        }
        virtual bool B_is_allowed(const struct ggml_tensor *B) {
            // TODO @ mettre dans un fils!
            // TODO tester avec le type_bf16 de ce tenseur maintenant qu'il est mis
            if (B->type != GGML_TYPE_F32) return false;
            if (ggml_is_transposed(B)) return false;
            return true;
        }
        virtual bool A_is_allowed(const struct ggml_tensor *A) = 0;

        virtual op* inst(const struct ggml_tensor *op) {
            return this; // pas de new ici...
        }

        // - ? gestion format des poids / choix du tenseur?
        static void set(const struct ggml_tensor *ggml_op, op* bf16_op) {
            // normalement a ne pas faire ;)
            const_cast<struct ggml_tensor *>(ggml_op)->bf16_op = bf16_op;
        }
        static inline op* get(struct ggml_tensor *ggml_op) {
            return (op*) ggml_op->bf16_op;
        }
        // - compute
        virtual void exec(struct ggml_tensor *op) const = 0;
    };

    // les ops possibles:
    // - avec les tenseurs bf16 d'origine!
    //op* op_jart_1 = nullptr;  // avec m<6
    //op* op_jart_2 = nullptr;  // avec m>6
    static std::list<op*> matmul_ops;
}

// tenseur:
//  - liste de supported?
// op
//  - liste de supported dans l'ordre de priorité

// cas a traiter par ordre d'importance:
//  - bf16 (celui actuel avec nouvelle "cobception)
//     bf16_32x1 N<=6 ...
//     bf16_32x1 N>6
//     bf16_2x16  (pour voir sa vitesse et tester les remaping)
//  - fp8 =>
//    E4M3_2x16 / sans scale pour tester la vitesse
//    E4M3_2x16 / avec scale + avec ou sans subnormal pour tester la qualité
//    ...
// - bf16:
//    bf16_8x8  => voir si on gagne sans reformatage (kv_cache...)
//    bf16_8x8  => avec reformatage (pour comparer a 2x16)


//#############################################################

// gestion de Matrice (en attandant mieux)
// template<typename T, int K, int M>  // => stockage par block!
// @ mettre au propre: K<>m M<>n
template<typename T, size_t K=1, size_t M=1>
class Matrice {
public:
    inline auto DIM1() const { return m_m; }
    inline auto DIM2() const { return m_n; }
    inline auto LD()   const { return m_l; }

    const std::size_t m_m;  // m contigue
    const std::size_t m_n;
    const std::size_t m_l;  // nb elements pour passer a la colonne (bloc) suivante.
    T* m_values;  // suivant le format ca peut-etre une copie!

    inline Matrice(T* v, std::size_t m, std::size_t n, std::size_t l): m_m(m), m_n(n), m_l((K*M==1)?l:m*K*M), m_values(v) {
        static_assert(K>0);
        static_assert(M>0);
    }

    // un buffer avec le meme format que le tenseur
    inline Matrice(const void* data, struct ggml_tensor * t):
            m_m(t->ne[0]), m_n(t->ne[1]),
            m_l((t->nb[1]/t->nb[0])),
            m_values((T*)(data))
    {
        static_assert(K==1);
        static_assert(M==1);
        GGML_ASSERT(t->ne[2]==1);
        GGML_ASSERT(t->ne[3]==1);
    }

    //inline Matrice(struct ggml_tensor * t): m_m(t->ne[0]), m_n(t->ne[1]), m_l((t->nb[1]/t->nb[0])*K*M), m_values((T*)(t->data)) {
    inline Matrice(struct ggml_tensor * t):
            m_m(t->ne[0]), m_n(t->ne[1]),
            m_l((K*M==1)?(t->nb[1]/t->nb[0]):t->ne[0]*M),
            m_values((T*)(t->data))
    {
        static_assert(K>0);
        static_assert(M>0);
        GGML_ASSERT(t->ne[2]==1);
        GGML_ASSERT(t->ne[3]==1);
        //GGML_ASSERT((sizeof(T)*m_m*m_n)<=ggml_nbytes(t));
        // std::cout <<" Matrice: "<< ":" <<m_m<<"/"<<m_n<<"/"<<m_l <<" | "<< t << std::endl;
    }

    inline Matrice(struct ggml_tensor * t, std::size_t i, std::size_t j): m_m(t->ne[0]), m_n(t->ne[1]), m_l(t->nb[1]/t->nb[0]) { //, m_values((T*)add_byte(t->data,)) {
        // pas utilisable pour les empilements de matrice non native!
        static_assert(K==1);
        static_assert(M==1);
        m_values = (T*) add_byte(t->data, i*t->nb[2]+j*t->nb[3]);
    }

    inline T& operator()(size_t i, size_t j) {
        const auto i0 = i%K; const auto i1 = i/K;
        const auto j0 = j%M; const auto j1 = j/M;
        return m_values[j1*m_l+j0*K+i1*K*M+i0];
        // return m_values[j*m_l+i];
    }
    inline const T operator()(size_t i, size_t j) const {
        const auto i0 = i%K; const auto i1 = i/K;
        const auto j0 = j%M; const auto j1 = j/M;
        return m_values[j1*m_l+j0*K+i1*K*M+i0];
        // return m_values[j*m_l+i];
    }

    inline T* addr(size_t i, size_t j) {
        if constexpr(K>1 || M>1) {
            // pas une bonne idée de taper en dehors d'un bloc complet,
            //GGML_ASSERT(i<m_m);
            //GGML_ASSERT(j<m_n);
            //GGML_ASSERT(j%M==0);
            //GGML_ASSERT(i%K==0);
            //GGML_ASSERT(j%M==0);
            const auto i1 = i/K;
            const auto j1 = j/M;
            return m_values+j1*m_l+i1*K*M;
        } else {
            return m_values+j*m_l+i; // [j1*m_l+j0*K+i1*K*M+i0]
        }
    }
    inline const T* addr(size_t i, size_t j) const {
        if constexpr(K>1 || M>1) {
            // pas une bonne idée de taper en dehors d'un bloc complet,
            //GGML_ASSERT(i%K==0);
            //GGML_ASSERT(j%M==0);
            const auto i1 = i/K;
            const auto j1 = j/M;
            return m_values+j1*m_l+i1*K*M;
        } else {
            return m_values+j*m_l+i; // [j1*m_l+j0*K+i1*K*M+i0]
        }
    }

    static bool valid(const struct ggml_tensor * t) {
        // @ priorie non transposé suffit aujourd'hui!
        return (!ggml_is_transposed(t)) && is<T>(t);
        //return ggml_is_contiguous(t) && is<T>(t); // && t->ne[2]==1 && t->ne[3]==1;
    }
};



//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------
// => implemetation des OPs



//------------------------------------------------------------------------------------
// => celui de JART!
// cas K%32 = 0
// - fp32: => bf16_x32
static inline auto load(const fp32_t *X) {
    auto x1 = _mm512_loadu_ps(X);
    auto x2 = _mm512_loadu_ps(X+16);
    return _mm512_cvtne2ps_pbh(x2,x1);
}

// - fp16: => bf16_x32
static inline auto load(const bf16_t *X) {
    return (__m512bh) _mm512_loadu_epi16(X);
}

// - fp8
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

static inline auto madd(const __m512bh& A, const __m512bh& B, const __m512& C) {
    return _mm512_dpbf16_ps(C, A, B);
}

// y appliquer un facteur de correction si besoin
template<bool SCALE=false>
static inline float hsum(__m512 x, fp32_t scale=1) {
    if constexpr (SCALE) {
        return scale*_mm512_reduce_add_ps(x);
    } else {
        return _mm512_reduce_add_ps(x);
    }
}

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
// p'etre un "masque" A(quantisé>bf16)/B(fp32>bf16)
enum class ACTION {
    NONE,
    STORE,
    LOAD
    // + ACC / C config!
};

// TODO: ajouter un facteur de correction pour C...
template<size_t M, size_t N, ACTION ACT=ACTION::NONE, bool ACC=false, typename TA, typename TB, typename TC>
static void gemm(const TA *pA, const TB *pB, TC *pC, std::size_t lda, std::size_t ldb, std::size_t ldc, std::size_t K, bf16_t *pB_=nullptr, std::size_t ldb_=0) {
    constexpr int K0 = 32; // 32 bf16 !
    static_assert(N>0);
    static_assert(M>0);
    // K%32 == 0!!
    // A[?,K+:lda]
    // B[?,K+:ldb]
    // C[?,ldc]
    __m512   C[M][N];
    __m512bh A[M];
#pragma GCC unroll N
    for(size_t j=0; j<N; j++) {
#pragma GCC unroll M
        for(size_t i=0; i<M; i++) {
            C[i][j] = _mm512_setzero_ps();
        }
    }
    //     #pragma GCC unroll K1 => voir si c'est mieux.
    //#pragma GCC unroll 4 //  ne change pas grand chose
    for (std::size_t k=0; k<K; k+=K0) {
#pragma GCC unroll M
        for(size_t i=0; i<M; i++) {
            A[i] = load(pA+i*lda+k);
        }
#pragma GCC unroll N
        for(size_t j=0; j<N; j++) {
            __m512bh B;
            // gestion d'un cache pour B
            if constexpr(ACT!=ACTION::LOAD) B = load(pB+j*ldb+k);
            if constexpr(ACT==ACTION::LOAD) B = load(pB_+j*ldb_+k);
#pragma GCC unroll M
            for(size_t i=0; i<M; i++) {
                C[i][j] = madd(A[i], B, C[i][j]);
            }
            if constexpr(ACT==ACTION::STORE) store(pB_+j*ldb_+k, B);
        }
    }

    // reduce and store C res.
#pragma GCC unroll N
    for(size_t j=0; j<N; j++) {
#pragma GCC unroll N
        for(size_t i=0; i<M; i++) {
            if constexpr (ACC) {
                pC[i+j*ldc] += hsum(C[i][j]); // TODO: * scal
            } else {
                pC[i+j*ldc] = hsum(C[i][j]); // TODO: * scal
            }
        }
    }
}

template<size_t M, size_t N, ACTION ACT=ACTION::NONE, bool ACC=false, typename TA, typename TB, typename TC>
static void sgemm_512_bloc(TA* A, TB* B, TC* C, size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc, bf16_t* B_, size_t ldb_) {
    GGML_ASSERT(m<=M);
    GGML_ASSERT(n<=N);

    // choix du kernel:
    if ((M==m) && (N==n)) { // seul cas traité pour l'instant
        gemm<M,N,ACT,ACC>(A, B, C, lda, ldb, ldc, k, B_, ldb_);
        return;
    }
    if constexpr (M>1) { // arret de la recursion
        if (M>m) {
            sgemm_512_bloc<M-1,N,ACT,ACC>(A,B,C,m,n,k,lda,ldb,ldc, B_, ldb_);
        }
    }
    if constexpr (N>1) { // arret de la recursion
        if (M==m && N>n) {
            sgemm_512_bloc<M,N-1,ACT,ACC>(A,B,C,m,n,k,lda,ldb,ldc, B_, ldb_);
        }
    }
}

template<size_t M1, size_t N1, size_t M0, size_t N0, size_t K0=1024, typename TA, typename TB, typename TC>
static inline void sgemm_512_bloc(const Matrice<TA>& A, const Matrice<TB>& B, Matrice<TC>& C, size_t I0, size_t J0, bf16_t* B_) {
    const size_t IN = std::min(C.DIM1(), I0+M1*M0);
    const size_t JN = std::min(C.DIM2(), J0+N1*N0);
    const auto KN = A.DIM1(); // == B.DIM1()

    if (B_) {
        for (size_t k=0; k<KN; k+=K0) {
            const auto _K = std::min(K0,KN-k);
            for (size_t j=J0; j<JN; j+=N0) {
                const auto _N = std::min(N0,JN-j);
                if (k==0) {
                    sgemm_512_bloc<M0,N0,ACTION::STORE,false>(A.addr(0,I0),B.addr(0,j),C.addr(I0,j),std::min(M0,IN-I0),_N,_K, A.LD(),B.LD(),C.LD(), B_, K0);
                } else {
                    sgemm_512_bloc<M0,N0,ACTION::STORE,true>(A.addr(k,I0),B.addr(k,j),C.addr(I0,j),std::min(M0,IN-I0),_N,_K, A.LD(),B.LD(),C.LD(), B_, K0);
                }
                if (I0+M0<IN)
                    for (size_t i=I0+M0; i<IN; i+=M0) {
                        const auto _M = std::min(M0,IN-i);
                        if (k==0) {
                            sgemm_512_bloc<M0,N0,ACTION::LOAD,false>(A.addr(0,i),B.addr(0,j),C.addr(i,j),_M,_N,_K, A.LD(),B.LD(),C.LD(), B_, K0);
                        } else {
                            sgemm_512_bloc<M0,N0,ACTION::LOAD,true>(A.addr(k,i),B.addr(k,j),C.addr(i,j),_M,_N,_K, A.LD(),B.LD(),C.LD(), B_, K0);
                        }
                    }
            }
        }
    } else {
        for (size_t k=0; k<KN; k+=K0) {
            const auto _K = std::min(K0,KN-k);
            for (size_t j=J0; j<JN; j+=N0) {
                const auto _N = std::min(N0,JN-j);
                for (size_t i=I0; i<IN; i+=M0) {
                    const auto _M = std::min(M0,IN-i);
                    if (k==0) {
                        sgemm_512_bloc<M0,N0,ACTION::NONE,false>(A.addr(0,i),B.addr(0,j),C.addr(i,j),_M,_N,_K, A.LD(),B.LD(),C.LD(), B_, K0);
                    } else {
                        sgemm_512_bloc<M0,N0,ACTION::NONE,true>(A.addr(k,i),B.addr(k,j),C.addr(i,j),_M,_N,_K, A.LD(),B.LD(),C.LD(), B_, K0);
                    }
                }
            }
        }
    }
}

//----------------------------------------------
// cas block de BF16:
namespace ggml::bf16::op_matmul {
    class bf16_2x16: public ggml::backend::bf16::op {
        // OK c'est prometeur on sature la memoire avec 2 threads!
        // Pour aller plus vite il va faloir gerer les caches!
        //  - A => en L2
        //  - B => en L1
        // => il fait gerer des K1
        // => parcourir M2 bloc
        // => parcourir tous les B
        // => passer a la suite...
        //   - K0 : dot2       => 2
        //   - M0 : simd       => 16
        //   - N0 : registres  => 16?
        //   - K1 : B en L1 (+conv bf16)  => 32k: 1024/2=512
        //   - M1 : A en L2               => 1Mo: 32 (bf16) / 64 (fp8) (512xNbThread)
        //   - N1 : B en L3 (fp32) / C en L3 (?)  => 16 Mo  => 256 (4096 fp32)???
        //   - K  : tous les K
        //   - M  : tous les M  \ dispache par coeurs
        //   - N  : tous les N  /
        // Note: reduction sur K ...
        //  https://passlab.github.io/Examples/contents/Chap_data_environment/9_Reduction.html#user-defined-reduction
        void exec(struct ggml_tensor *op) const override {
#ifdef DO_TIMING
            mesure time; time.start();
#endif
            // normalement ici les type sont deja controlé
            const Matrice<bf16_t,2,16> A(op->src[0]);
            const Matrice<fp32_t>      B(op->src[1]);
            Matrice<fp32_t>            C(op);
            mul_mat(A, B, C);
#ifdef DO_TIMING
            auto dt = time.end();
            std::cout << " bf16_2x16> " <<op->op<<"("<<log_srcs(op)<<" => "<<op<<"): "<< op->name << " => "
                    << dt*1000000 << " us / "
                    << (1e-9*2*op->ne[0]*op->ne[1]*op->src[1]->ne[0])/dt << " GFlops/s"
                    << std::endl;
#endif
        }

        // voir si on fait plusieurs cas 16xN  32xN...
        //bool B_is_allowed(const struct ggml_tensor *B) override {
        //    if (B->ne[1]>6) return false;
        //    return ggml_bf16_op_matmul::B_is_allowed(B);
        //}
        bool A_is_allowed(const struct ggml_tensor *A) override {
            auto a = ggml::backend::bf16::tensor::GET(A);
            if(!a && A->view_src) {
                // une vue => pas "encore" supporté!
                //  il y a les kv cache... => mais pas re-formatable de toute facon!!!
                //llama.cpp/ggml-bf16.cpp@670: type de A non defini: VIEW/v-39[64:128:8:1/2:256:32768:262144]@bf16
                //  => VIEW: NONE/cache_v_l39[131072:1:1:1/2:262144:262144:262144]@bf16/0x31c50c0
                //llama.cpp/ggml-bf16.cpp@670: type de A non defini: VIEW/k-39[128:64:8:1/2:2048:256:2048]@bf16
                //  => VIEW: NONE/cache_k_l39[131072:1:1:1/2:262144:262144:262144]@bf16/0x31c50c0
                return false;
            }
            GGML_ASSERT(a!=nullptr);
            // on sais deja:
            if (a->type == ggml::backend::bf16::TYPE::BF16_2x16) return true;
            return false;
        }

        // le bloc de bas niveau
        // TODO: gerer des Tensor / BlocTensor et pas des Type* !!!
        // ca evite de passer ld[a,b,c] et permet de gerer correctement les different bloc!
        //  => K0/M0 sont alors fixé pour A!
        //  et gerer les conversions / acces vecteurs ...
        //  load<bf16_32> / load<bf16_2x16> / load<bf16_t,32>...
        // ??? les BlocTensor n'ont p'etre pas trop d'acces possible load_v/load_s ...
        //  ou load(out, i, j);
        // TODO: changer M1 => "M1/M0"
        template<size_t M1, size_t N0, typename TA, typename TB, typename TC>
        static void gemm(const TA *pA, const TB *pB, TC *pC, std::size_t lda, std::size_t ldb, std::size_t ldc, std::size_t K2) {
            // lda: comment passer de A[k,i] => A[k,i+M1]
            // ldb: comment passer de B[k,j] => B[k,j+1]
            // ldc: comment passer de C[i,j] => C[i,j+1]
            constexpr int K0 =  2; //
            constexpr int M0 = 16; // 16 FP32 => 1 AVX512!
            constexpr int K1 =  8; // des blocks de 4 (8/2) pour ameloré les lecture de B!
            static_assert(M1>0);
            static_assert(N0>0);
            static_assert(M1%M0 == 0);
            GGML_ASSERT(K2%K1 == 0);

            // K%32 == 0!!
            // A[?,K+:lda]
            // B[?,K+:ldb]
            // C[?,ldc]
            __m512   C[M1/M0][N0];    // m512   == fp32[M0]
            __m512bh A[K1/K0][M1/M0]; // m512bh == bf16[M0][K0]

            //std::cout << "  - 0" << std::endl;
#pragma GCC unroll N0
            for(size_t j=0; j<N0; j++) {
//#pragma GCC unroll (M1/M0)
#pragma GCC unroll M1
                for(size_t i=0; i<(M1/M0); i++) {
                    C[i][j] = _mm512_setzero_ps();
                }
            }

            for (size_t k2=0; k2<K2; k2+=K1) { // de 8 en 8 ...
                // chargement de A
#pragma GCC unroll K1
                for (size_t k1=0; k1<K1/K0; ++k1) {  // [0..3]
#pragma GCC unroll M1
                    for (size_t i1 = 0; i1<M1/M0; ++i1) {  // [1]
                        A[k1][i1] = load(pA + i1*lda + k2*M0 + k1*M0*K0); // lda == K2*M0 ...
                    }
                }
#pragma GCC unroll N0
                for (size_t j=0; j<N0; ++j) {  // [0..~16]
                    // on charge K1 valeur de B
                    __m128bh B = _mm256_cvtneps_pbh(_mm256_loadu_ps(pB+j*ldb+k2));

#pragma GCC unroll K1
                    for (size_t k1=0; k1<K1/K0; ++k1) {  // [0..4]
                        auto _B = _mm512_broadcastd_2pbh(B);
                        B = _mm_shiftl_2pbh(B);
#pragma GCC unroll M1
                        for (size_t i1=0; i1<M1/M0; ++i1) {  // [1]
                            C[i1][j] = madd(A[k1][i1], _B, C[i1][j]);
                        }
                    }
                }
            }

            // ecriture de C...
#pragma GCC unroll N0
            for(size_t j=0; j<N0; j++) {
//#pragma GCC unroll (M1/M0)
#pragma GCC unroll M1
                for (size_t i1=0; i1<M1/M0; ++i1) {
                    store(pC+j*ldc+i1*M0, C[i1][j]);
                }
            }
        }

        template<size_t M1, size_t N0, size_t M0, size_t K0>
        inline void sgemm_bloc(const Matrice<bf16_t,K0,M0>& A, const Matrice<fp32_t>& B, Matrice<fp32_t>& C, size_t i, size_t j, size_t N, size_t K) const {
            static_assert(N0<=16);
            switch (N) {
                case 16: gemm<M1*M0,16>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K); break;
                case 15: gemm<M1*M0,15>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K); break;
                case 14: gemm<M1*M0,14>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K); break;
                case 13: gemm<M1*M0,13>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K); break;
                case 12: gemm<M1*M0,12>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K); break;
                case 11: gemm<M1*M0,11>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K); break;
                case 10: gemm<M1*M0,10>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K); break;
                case  9: gemm<M1*M0, 9>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K); break;
                case  8: gemm<M1*M0, 8>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K); break;
                case  7: gemm<M1*M0, 7>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K); break;
                case  6: gemm<M1*M0, 6>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K); break;
                case  5: gemm<M1*M0, 5>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K); break;
                case  4: gemm<M1*M0, 4>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K); break;
                case  3: gemm<M1*M0, 3>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K); break;
                case  2: gemm<M1*M0, 2>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K); break;
                case  1: gemm<M1*M0, 1>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K); break;
                default: break;
            }
            // std::cout << "    N="<<N<<"/"<<N0<< std::endl;
            /*
            if (N==N0) {
                // calcule
                gemm<M1*M0,N0>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K);
                return;
            }
            if constexpr (N0>1) { // arret de la recursion
                sgemm_bloc<M1,N0-1,M0,K0>(A, B, C, i, j, N, K);
            }
            */
        }

        template<size_t K0, size_t M0>
        void mul_mat(const Matrice<bf16_t,K0,M0>& A, const Matrice<fp32_t>& B, Matrice<fp32_t>& C) const {
            static_assert(K0==2);
            static_assert(M0==16);
            const auto m = C.DIM1(); // == A.DIM2()
            const auto n = C.DIM2(); // == B.DIM2()
            const auto k = A.DIM1(); // == B.DIM1()
            // K0 = 2; M0 = 16; !!!
            constexpr size_t N0 = 16;
            constexpr size_t M1 =  1;
            constexpr size_t M2 =  2;

            //#pragma omp parallel for private(B_cache) schedule(guided)
            // bool premier = true;
            //#pragma omp parallel for private(premier) private(B_cache) schedule(guided)
#pragma omp parallel for schedule(guided)
            for (size_t i=0; i<m; i+=M2*M1*M0) {
                for (size_t j=0; j<n; j+=N0) {
                    const auto N = std::min(n-j,N0);
                    // TODO: premier => mettre B<bf16> en cache
#pragma GCC unroll 8
                    for (size_t i2=0; i2<M2*M1*M0; i2+=M1*M0) {
                        sgemm_bloc<M1,N0,M0,K0>(A, B, C, i+i2, j, N, k);
                    }
                }
            }
        }

    };

}

//----------------------------------------------
// cas sans bloc de BF16:
class ggml_bf16_op_matmul : public ggml::backend::bf16::op {
public:
    void exec(struct ggml_tensor *op) const override {
#ifdef DO_TIMING
        mesure time; time.start();
#endif
        const auto src0 = op->src[0];
        const auto src1 = op->src[1];
        auto dst  = op;

        // broadcast factors
        const auto r2 = src1->ne[2]/src0->ne[2];
        const auto r3 = src1->ne[3]/src0->ne[3];

        for (int64_t i13 = 0; i13 < src1->ne[3]; i13++) {
            for (int64_t i12 = 0; i12 < src1->ne[2]; i12++) {
                const auto i03 = i13/r3;
                const auto i02 = i12/r2;

                const Matrice<bf16_t> A(src0,i02,i03);
                const Matrice<fp32_t> B(src1,i12,i13);
                Matrice<fp32_t> C(dst,i12,i13);
                mul_mat(A, B, C);
            }}
#ifdef DO_TIMING
        auto dt = time.end();
        std::cout << " > " <<op->op<<"("<<log_srcs(op)<<" => "<<op<<"): "<< op->name << " => "
                << dt*1000000 << " us / "
                << (1e-9*2*src1->ne[3]*src1->ne[2]*op->ne[0]*op->ne[1]*src1->ne[0])/dt << " GFlops/s"
                << std::endl;
#endif
    }
    //  - bf16+fp32=fp32
    virtual void mul_mat(const Matrice<bf16_t>& A, const Matrice<fp32_t>& B, Matrice<fp32_t>& C) const = 0;
};

class ggml_bf16_op_matmul_1 : public ggml_bf16_op_matmul {
public:
    //if (B/C->ne[1]>6 );
    bool B_is_allowed(const struct ggml_tensor *B) override {
        if (B->ne[1]>4) return false;
        return ggml_bf16_op_matmul::B_is_allowed(B);
    }
    bool A_is_allowed(const struct ggml_tensor *A) override {
        auto a = ggml::backend::bf16::tensor::GET(A);
        if(!a && A->view_src) {
            // une vue => pas "encore" supporté
            return false;
        }
        GGML_ASSERT(a!=nullptr);
        // on sais deja:
        if (a->type == ggml::backend::bf16::TYPE::BF16_32x1) return true;
        // y a t'il d'autre cas compatible?
        return false;
    }
    void mul_mat(const Matrice<bf16_t>& A, const Matrice<fp32_t>& B, Matrice<fp32_t>& C) const override {
        const auto m = C.DIM1(); // == A.DIM2()
        const auto n = C.DIM2(); // == B.DIM2()
        const auto k = A.DIM1(); // == B.DIM1()
        GGML_ASSERT(A.LD()>=k);
        GGML_ASSERT(B.LD()>=k);
        GGML_ASSERT(C.LD()>=m);
        // ici k<=6 ...
        constexpr size_t M0 = 8; //4;
        constexpr size_t N0 = 3; //6;
        constexpr size_t M1 = 4;
        constexpr size_t K0 = 2560; // 5120; //2048+512; // 4096+1024;
        //static thread_local bf16_t B_cache[N0*K0];
        bf16_t B_cache[N0*K0];

#pragma omp parallel for private(B_cache) schedule(guided)
        //#pragma omp parallel for schedule(guided)
        for (size_t i=0; i<m; i+=M1*M0) {
            //sgemm_512_bloc<M1,1,M0,N0,K0>(A, B, C, i, 0, nullptr);
            sgemm_512_bloc<M1,1,M0,N0,K0>(A, B, C, i, 0, B_cache);
        }
    }
};

class ggml_bf16_op_matmul_2 : public ggml_bf16_op_matmul {
public:
    // la derniere chance!
    bool A_is_allowed(const struct ggml_tensor *A) override {
        auto a = ggml::backend::bf16::tensor::GET(A);
        if(!a && A->view_src) {
            // une vue => pas "encore" supporté
            return false;
        }
        GGML_ASSERT(a!=nullptr);
        // on sais deja:
        if (a->type == ggml::backend::bf16::TYPE::BF16_32x1) return true;
        // y a t'il d'autre cas compatible?
        return false;
    }
    void mul_mat(const Matrice<bf16_t>& A, const Matrice<fp32_t>& B, Matrice<fp32_t>& C) const override {
        const auto m = C.DIM1(); // == A.DIM2()
        const auto n = C.DIM2(); // == B.DIM2()
        const auto k = A.DIM1(); // == B.DIM1()
        GGML_ASSERT(A.LD()>=k);
        GGML_ASSERT(B.LD()>=k);
        GGML_ASSERT(C.LD()>=m);

        // la taille des plus grand blocs.
        constexpr size_t M0 = 5;
        constexpr size_t N0 = 5;
        constexpr size_t M1 = 8;
        constexpr size_t N1 = 4;
        constexpr size_t K0 = 5120; //2560; //4096;
        bf16_t B_cache[N0*K0];

        // schedule(dynamique)
#pragma omp parallel for collapse(2) private(B_cache) schedule(guided)
        for (size_t i=0; i<m; i+=M1*M0) {
            for (size_t j=0; j<n; j+=N1*N0) {
                sgemm_512_bloc<M1,N1,M0,N0,K0>(A, B, C, i, j, B_cache);
            }
        }
    }
};

//////////////////////////////////////////////////////////////////////////////////
// l'init du backend:
//------------------------------------------------------------------------------------
// le context definissant le backend
// - les tenseur??? comment on gere les tenseurs du backend?
//namespace ggml::backend::bf16::tensor {
//}
// - le buffer
namespace ggml::backend::bf16::buffer {
    // TODO: le buffer de ce backend
    //  => comme pour les autre C => class context;
    static constexpr std::size_t TENSOR_ALIGNMENT = 64;
    class context {
    public:
        context(size_t size) : m_size(size) {
            m_data = new (std::align_val_t(TENSOR_ALIGNMENT)) uint8_t[m_size];
        }
        ~context() {
            delete[] m_data;
        }
        inline std::size_t get_size() {
            return m_size;
        }
        // Note[JPP]: voir si on l'initialise dans l'init du backend...
        // les tenseurs qui sont converti en FP8 / Q8 ?
        // - https://huggingface.co/neuralmagic/Mistral-Nemo-Instruct-2407-FP8/tree/main?show_file_info=model-00003-of-00003.safetensors
        // - https://huggingface.co/lmstudio-community/Mistral-Nemo-Instruct-2407-GGUF/tree/main?show_file_info=Mistral-Nemo-Instruct-2407-Q8_0.gguf
        static constexpr std::list<std::string> LIST_BF16_2x16() { return {
                "fn_down.weight",
                "fn_gate.weight",
                "fn_up.weight",
                "ttn_k.weight",
                "ttn_q.weight",
                "ttn_v.weight",
                "ttn_output.weight",
        };}
        /*
blk.0.ffn_down.weight   [14 336, 5 120]     Q8_0
blk.0.ffn_gate.weight   [5 120, 14 336]     Q8_0
blk.0.ffn_up.weight     [5 120, 14 336]     Q8_0
blk.0.attn_k.weight     [5 120, 1 024]  Q8_0
blk.0.attn_q.weight     [5 120, 4 096]  Q8_0
blk.0.attn_v.weight     [5 120, 1 024]  Q8_0
blk.0.attn_output.weight    [4 096, 5 120]  Q8_0
blk.0.ffn_norm.weight   [5 120]     F32
blk.0.attn_norm.weight  [5 120]     F32
         */
        inline ggml::backend::bf16::tensor* get_tensor_type(ggml_tensor * tensor) {
            if (ggml_is_transposed(tensor)) return nullptr;
            ggml::backend::bf16::tensor* t = nullptr;
            if ( (tensor->flags&GGML_TENSOR_FLAG_WEIGHTS) == GGML_TENSOR_FLAG_WEIGHTS) {
                const std::string name = ggml_get_name(tensor);
                // les poids ... ils peuvent supporter plus de transformations!
                //   la transformation ser faite par le set_tensor
                //   ceux qui sont eligible sont restreint a ceux qui servent au matmul!
                if (tensor->type == GGML_TYPE_BF16) {
                    // K0=2 / M0=16:
                    if ((tensor->ne[0] % 2 == 0) && (tensor->ne[1] % 16 == 0) && (tensor->ne[2] == 1) && (tensor->ne[3] == 1)) {
                        for (auto patern: LIST_BF16_2x16()) {
                            if (name.find(patern) != std::string::npos) {
                                //std::cout << "transforme<bf16_2x16>: " <<tensor->name<< tensor << std::endl;
                                return &ggml::backend::bf16::tensor_bf16_2x16_t;
                            }
                        }
                    }
                }
                //std::cout << "transforme<bf16_2x16> non supporté: " <<tensor->op<<"<"<<tensor->name<<"> "<< tensor << std::endl;
            }
            // OK pas de transformation possible => les format "natif"
            switch (tensor->type) {
            case GGML_TYPE_BF16 :
                if (tensor->ne[0] % 32 == 0) {
                    t=&ggml::backend::bf16::tensor_bf16_32x1_t;
                }
                // std::cout << "Tenseur<"<<tensor->name<<">: bf16_32x1" << std::endl;
                break;
            case GGML_TYPE_F32 :
                //  il faut verifier en fonction de type de produit le format a utiliser...
                // + tensor_fp32_32x1_t
                // + tensor_fp32_8x1_t
                t=&ggml::backend::bf16::tensor_fp32_t;
                break;
            }
            return t;
        }
        // l'interface: ggml_backend_buffer_i
        inline const char * get_name(void){ return "BF16"; }
        inline void *       get_base(void) { return m_data; }
        inline void         init_tensor(ggml_tensor * tensor) {
            // on configurer tous les tenseur tel que l'on veux qu'il soit:
            auto t = ggml::backend::bf16::tensor::GET(tensor);
            if (t == nullptr) {
                // pas encore vu => on va le configurer:
                t = get_tensor_type(tensor);
                // OK on fixe le type !
                if (t) {
                    ggml::backend::bf16::tensor::SET(tensor, t);
                } else {
                    ggml::backend::bf16::tensor::SET(tensor, &ggml::backend::bf16::tensor_non_supporte_t);
                }
            } else {
                std::cout << "re-init_tensor:" <<tensor->op<< tensor << std::endl;
            }
        }
        // inline void load_tensor(struct ggml_tensor * tensor, File, offset);
        inline void set_tensor(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
            // TODO: traiter tous les types du backend!!!
            if (ggml::backend::bf16::tensor::GET(tensor)->type == ggml::backend::bf16::TYPE::BF16_2x16) {
                // if (tensor->type == GGML_TYPE_BF16) {
                // le seul cas pour l'instant supporté: une matrice de BF16 affecté en 1 bloc
                // std::cout << "set_tensor<"<<tensor->name<<">:" << offset << ":" << size <<"/"<<ggml_nbytes(tensor)<< std::endl;
                GGML_ASSERT(offset == 0);
                GGML_ASSERT(size == ggml_nbytes(tensor));
                GGML_ASSERT(tensor->type == GGML_TYPE_BF16);
                GGML_ASSERT(tensor->ne[2] == 1);
                GGML_ASSERT(tensor->ne[3] == 1);

                Matrice<bf16_t,2,16> A(tensor);
                const Matrice<bf16_t> B(data, tensor);
                for (int j=0; j<tensor->ne[1]; j++) {
                    for (int i=0; i<tensor->ne[0]; i++) {
                        //A(i,j) = *((bf16_t*)data[i*tensor->nb[0]+j*tensor->nb[1]]);
                        A(i,j) = B(i,j);
                    }
                }
                return;
            }
            memcpy((char *)tensor->data + offset, data, size);
        }
        inline void         get_tensor(const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size) {
            // @ revoir ca va dependre du type dans le backend!!!
            //std::cout << "get_tensor:" <<tensor->op<< tensor << std::endl;
            memcpy(data, (const char *)tensor->data + offset, size);
        }
        inline bool         cpy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
            // @ revoir ca va dependre du type dans le backend!!!
            if (ggml_backend_buffer_is_host(src->buffer)) {
                std::cout << "cpy_tensor:" << src <<"/"<< dst << std::endl;
                memcpy(dst->data, src->data, ggml_nbytes(src));
                return true;
            }
            return false;
        }
        inline void         clear(uint8_t value) {
            // mise a part mettre tout a 0 ... je pense pas qu'il puisse y avoir d'autre valeur...
            for (std::size_t i=0; i<m_size; ++i) m_data[i]= value;
        }
        inline void         reset() {}

    private:
        // les données du "context == buffer"
        uint8_t * m_data;
        const std::size_t m_size;
    };

    // les wrapper:
    static inline context* ctx(ggml_backend_buffer_t buffer) { return (context*) buffer->context; }
    static GGML_CALL const char * get_name(ggml_backend_buffer_t buffer){
        return ctx(buffer)->get_name();
    }
    static GGML_CALL void free_buffer(ggml_backend_buffer_t buffer){
        delete ctx(buffer);
    }
    static GGML_CALL void * get_base(ggml_backend_buffer_t buffer) {
        return ctx(buffer)->get_base();
    }
    static GGML_CALL void init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
        ctx(buffer)->init_tensor(tensor);
    }
    static GGML_CALL void set_tensor(ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
        ctx(buffer)->set_tensor(tensor, data, offset, size);
    }
    static GGML_CALL void get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size){
        ctx(buffer)->get_tensor(tensor, data, offset, size);
    }
    static GGML_CALL bool cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst){
        return ctx(buffer)->cpy_tensor(src, dst);
    }
    static GGML_CALL void clear(ggml_backend_buffer_t buffer, uint8_t value) {
        return ctx(buffer)->clear(value);
    }
    static GGML_CALL void reset(ggml_backend_buffer_t buffer) {
        return ctx(buffer)->reset();
    }

    static struct ggml_backend_buffer_i interface = {
            /* .get_name        = */ get_name,
            /* .free_buffer     = */ free_buffer,
            /* .get_base        = */ get_base,
            /* .init_tensor     = */ init_tensor,
            /* .set_tensor      = */ set_tensor,
            /* .get_tensor      = */ get_tensor,
            /* .cpy_tensor      = */ cpy_tensor,
            /* .clear           = */ clear,
            /* .reset           = */ reset,
    };
}

// - le buffer_type
namespace ggml::backend::bf16::buffer_type {

    class context {
    public:
        inline const char * get_name() { return "BF16"; }
        // inline ggml_backend_buffer_t alloc_buffer(size_t size) { return nullptr; }
        inline size_t get_alignment(){
            return ggml::backend::bf16::buffer::TENSOR_ALIGNMENT; // @ voir quoi mettre: 512 bits?
        }
        inline size_t get_max_size(){
            return (size_t)64*1024*1024*1024; // la taille de la RAM/GGT/VRAM ???
        }
        inline size_t get_alloc_size(const struct ggml_tensor * tensor) {
            // TODO: gerer les cas reformaté.
            //  probleme: on n'a pas encore sont "vrai" type!
            //  - poids pour les matmul => en fct du "type" cible
            //if (ggml::bf16::tensor::get(tensor)->type == ggml::bf16::TYPE::BF16_2x16) {
            //    return 2*tensor->ne[0]*tensor->ne[1];
            //}
            return ggml_nbytes(tensor);
        }
        inline bool is_host() {
            return true;
        }
    };

    // Les wrapper:
    static inline context* ctx(ggml_backend_buffer_type_t buft) { return (context*) buft->context; }
    static GGML_CALL const char * get_name(ggml_backend_buffer_type_t buft) {
        return ctx(buft)->get_name();
    }
    static GGML_CALL ggml_backend_buffer_t alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
        auto buffer_ctx = new ggml::backend::bf16::buffer::context(size);
        return ggml_backend_buffer_init(buft, ggml::backend::bf16::buffer::interface, buffer_ctx , buffer_ctx->get_size());
        //return ctx(buft)->alloc_buffer(size);
    }
    static GGML_CALL size_t get_alignment(ggml_backend_buffer_type_t buft){
        return ctx(buft)->get_alignment();
    }
    static GGML_CALL size_t get_max_size(ggml_backend_buffer_type_t buft){
        return ctx(buft)->get_max_size();
    }
    static GGML_CALL GGML_CALL size_t get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor){
        return ctx(buft)->get_alloc_size(tensor);
    }
    static GGML_CALL bool is_host(ggml_backend_buffer_type_t buft){
        return ctx(buft)->is_host();
    }

    // ATTENTION il faut surement que ca soit static (pour ggml_backend_bf16_init / et 1 par device pour les buffer_type ???)
    inline ggml_backend_buffer_type_t get() {
        // il peut y en avoir 1 par device... mais tjs static!!
        static context ctx;
        static struct ggml_backend_buffer_type ggml_backend_buffer_type = {
                /* .iface = */ {
                        /* .get_name         = */ get_name,
                        /* .alloc_buffer     = */ alloc_buffer,
                        /* .get_alignment    = */ get_alignment,
                        /* .get_max_size     = */ get_max_size, // defaults to SIZE_MAX
                        /* .get_alloc_size   = */ get_alloc_size, // defaults to ggml_nbytes
                        /* .is_host          = */ is_host,
                },
                /* .context = */ &ctx,
        };
        return &ggml_backend_buffer_type;
    }
}

// - le backend
namespace ggml::backend::bf16 {

    // TODO: deplacer la class ggml_backend_bf16_context ici!
    //using context = ggml_backend_bf16_context;
    //class context {
    //};
    // - le context
    class context {
    public:

        ~context() {
            // y a t'il qq-chose a faire.
        }

        // le nom du backend
        inline const char * name() { return "BF16"; }

        // le type de buffer gere par ce backend
        inline ggml_backend_buffer_type_t get_default_buffer_type() {
            return ggml::backend::bf16::buffer_type::get();
        }

        // execution du graph contenant les OPs "supported"
        inline enum ggml_status graph_compute(struct ggml_cgraph * cgraph) {
            //std::cout << "." ;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                struct ggml_tensor * node = cgraph->nodes[i];

                switch (node->op) {
                case GGML_OP_MUL_MAT:
                    //std::cout << "RUN> " <<node->op<<"@"<<node->name<< " : "<<node->src[0]->name<<std::endl;
                    ggml::backend::bf16::op::get(node)->exec(node);
                    break;

                    //case GGML_OP_OUT_PROD:
                    //    ggml_backend_blas_out_prod(ctx, node);
                    //    break;

                    // y a ca dans backend OPENBLAS ... sais pas pourquoi.
                case GGML_OP_NONE:  // les poids ou les caches... => p'etre les traiter ici (init)
                    std::cout << "RUN> " <<node->op<<"@"<<node->name<<std::endl;
                    // => pas appelé!!!
                    break;
                case GGML_OP_RESHAPE:
                case GGML_OP_VIEW:
                case GGML_OP_PERMUTE:
                case GGML_OP_TRANSPOSE:
                    break;

                default:
                    fprintf(stderr, "%s: unsupported op %s\n", __func__, ggml_op_desc(node));
                    GGML_ASSERT(false);
                }
            }

            return GGML_STATUS_SUCCESS;
        }

        // doit indiqué si cette operation peut etre executé
        inline bool supports_op(const struct ggml_tensor * op) {
            static int test = 1;
            //auto op_base = ggml_bf16_op::get(op); // tjs null, le graph est tjs re-caculé
            // dispach des OPs
            //  - tous les tenseurs sont configuré (par l'init_buffer)
            //  - les poids sont converti si besoin (par set_buffer)
            // reste donc a selectioner l'OP compatible!
            if (op->op == GGML_OP_MUL_MAT) {
                // choisir la 1er OP possible!
                for (auto imp : ggml::backend::bf16::matmul_ops) {
                    // l'ordre est important...
                    if (imp->C_is_allowed(op) &&
                            imp->B_is_allowed(op->src[1]) &&
                            imp->A_is_allowed(op->src[0]))
                    {
                        auto bf16_op = imp->inst(op);
                        if (bf16_op) {
                            ggml::backend::bf16::op::set(op, bf16_op);
                            return true;
                        }
                    }
                }
                return false;
            } else if (op->op == GGML_OP_NONE) {
                // OK deplacé dans l'init, ca devrait etre bon
                if (! ggml::backend::bf16::tensor::GET(op)) {
                    std::cout << "ERREUR: tenseur non configuré: " <<op->op<<op<< std::endl;
                }
                // TODO: ca ne sert a rien mais on peu retourner true si le poid est pour nous
                return false;
            }
            // else if (op->op == GGML_OP_OUT_PROD) {
            //    std::cout << " > " <<op->op<<"("<<log_srcs(op)<<" => "<<op<<"): "<< op->name<<std::endl;
            //}
            //if (op_base==nullptr) {
            //std::cout << "TODO"<<(void*)op<<"/"<<op->bf16_op;
            //ggml_bf16_op::set(op, new ggml_bf16_op_notsupported());
            //std::cout << "/"<<op->bf16_op<<"/"<<ggml_bf16_op::get(op)<<"> " <<op->op<<"@"<<op->name<<" ("<<log_srcs(op)<<" => "<<op<<")"<<std::endl;
            //}
            return false;
        }

        // ???
        inline bool supports_buft(ggml_backend_buffer_type_t buft) {
            return ggml_backend_buft_is_host(buft);
        }

    private:
        // AUTRE_OP
    };

    static inline context * ctx(ggml_backend_t backend) { return (context *)backend->context; }

    // les methodes du backend => wrapper (elle sont codé dans la classe)
    static GGML_CALL const char * name(ggml_backend_t backend) {
        return ctx(backend)->name();
    }

    static GGML_CALL void free(ggml_backend_t backend) {
        //ggml_backend_bf16_context * ctx = (ggml_backend_bf16_context *)backend->context;
        delete ctx(backend);
        delete backend;
    }

    static GGML_CALL ggml_backend_buffer_type_t get_default_buffer_type(ggml_backend_t backend) {
        return ctx(backend)->get_default_buffer_type();
    }

    static GGML_CALL enum ggml_status graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
        return ctx(backend)->graph_compute(cgraph);
    }

    static GGML_CALL bool supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
        return ctx(backend)->supports_op(op);
    }

    static GGML_CALL bool supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
        return ctx(backend)->supports_buft(buft);
    }

    static struct ggml_backend_i interface = {
            /* .get_name                = */ name,
            /* .free                    = */ free,
            /* .get_default_buffer_type = */ get_default_buffer_type,
            /* .set_tensor_async        = */ nullptr,
            /* .get_tensor_async        = */ nullptr,
            /* .cpy_tensor_async        = */ nullptr,
            /* .synchronize             = */ nullptr,
            /* .graph_plan_create       = */ nullptr,
            /* .graph_plan_free         = */ nullptr,
            /* .graph_plan_update       = */ nullptr,
            /* .graph_plan_compute      = */ nullptr,
            /* .graph_compute           = */ graph_compute,
            /* .supports_op             = */ supports_op,
            /* .supports_buft           = */ supports_buft,  // si tout les entrée sont suporté l'op devrai lui etre assigné??? enfin pas sur!!
            /* .offload_op              = */ nullptr,
            /* .event_new               = */ nullptr,
            /* .event_free              = */ nullptr,
            /* .event_record            = */ nullptr,
            /* .event_wait              = */ nullptr,
            /* .event_synchronize       = */ nullptr,
    };

    static ggml_guid_t guid(void) {
        static ggml_guid guid = { 0xca, 0xfe, 0xde, 0xca, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01 };
        return &guid;
    }

    static void init() {
        static bool first = true;
        if (!first) return;
        first = false;

        std::string backend_config{getenv("GGML_USE_BACKEND_BF16")};
        // changer la config en fonction de l'option choisi:
        // backend_config == "bf16_32x1"
        // backend_config == "bf16_2x16"
        // backend_config == "E4M3_8x8_DAZ" // (denormals-are-zero)"
        // backend_config == "E4M3_8x8"     // process subnormal FP8 nomber on BF16 conversion
        // ...

        ggml::backend::bf16::matmul_ops.clear();
        // bloc de BF16 / FP8
        // TODO: if (backend_config == "bf16_2x16")
        ggml::backend::bf16::matmul_ops.push_back(new ggml::bf16::op_matmul::bf16_2x16);

        // les "JART" en dernier
        ggml::backend::bf16::matmul_ops.push_back(new ggml_bf16_op_matmul_1);
        ggml::backend::bf16::matmul_ops.push_back(new ggml_bf16_op_matmul_2);
    }

    static bool is_enable() {
        if (getenv("GGML_USE_BACKEND_BF16") == nullptr) return false;
        // voir ce qui peut conditioner le backend...
        //  cas llamafile ???
        return (X86_HAVE(FMA) && X86_HAVE(F16C) && X86_HAVE(AVX2) && X86_HAVE(AVX512F) && X86_HAVE(AVX512BW) && X86_HAVE(AVX512DQ) && X86_HAVE(AVX512VL) && X86_HAVE(AVX512_BF16));
    }

}

GGML_CALL ggml_backend_buffer_type_t ggml_backend_bf16_buffer_type(void) {
    if (!ggml::backend::bf16::is_enable()) return nullptr;
    return ggml::backend::bf16::buffer_type::get();
}

GGML_CALL ggml_backend_t ggml_backend_bf16_init(void) {
    if (!ggml::backend::bf16::is_enable()) return nullptr;

    // OK il faut le creer:
    ggml::backend::bf16::init();
    // auto context = new ggml_backend_bf16_context;
    auto backend = new ggml_backend {
        /* .guid      = */ ggml::backend::bf16::guid(),
        /* .interface = */ ggml::backend::bf16::interface,
        /* .context   = */ new ggml::backend::bf16::context,
    };
    return backend;
}
#else
GGML_CALL ggml_backend_t ggml_backend_bf16_init(void) {
    return nullptr;
}
GGML_CALL ggml_backend_buffer_type_t ggml_backend_bf16_buffer_type(void) {
    return nullptr;
}

#endif
