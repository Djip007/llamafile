//#define __x86_64__
/*
> build:
make clean
make -j16
make -j16 install PREFIX=/home/philou/LLM/usr/

// trace des tenseurs
GGML_USE_BACKEND_BF16=1 ./usr/bin/llamafile -m Mistral-Nemo-Instruct-2407.BF16.gguf -c 128 -n 1 -t 0 -p "[INST]bonjour a tu un nom. je ne sais pas comment t'appeler. Si tu n'en as pas je peux t'appeler TINTIN[/INST]"

TRACE> MUL_MAT@result_output (s0(output.weight)[5120:131072:1:1/2:10240:1342177280:1342177280]@bf16, s1(output.weight)[5120:128:1:1/4:20480:2621440:2621440]@f32 => [131072:128:1:1/4:524288:67108864:67108864]@f32)
TRACE> MUL_MAT@result_output (s0(output.weight)[5120:131072:1:1/2:10240:1342177280:1342177280]@bf16, s1(output.weight)[5120:1:1:1/4:20480:20480:20480]@f32 => [131072:1:1:1/4:524288:524288:524288]@f32)
TRACE> MUL_MAT@result_output (s0(output.weight)[5120:131072:1:1/2:10240:1342177280:1342177280]@bf16, s1(output.weight)[5120:1:1:1/4:20480:20480:20480]@f32 => [131072:1:1:1/4:524288:524288:524288]@f32)

> run: une fois implementé
GGML_USE_BACKEND_BF16=1 OMP_NUM_THREADS=8 ./usr/bin/llamafile -m Mistral-Nemo-Instruct-2407.BF16.gguf -c 128 -n 16 -t 0 -p "[INST]bonjour a tu un nom. je ne sais pas comment t'appeler. Si tu n'en as pas je peux t'appeler TINTIN[/INST]"
# GGML_USE_BACKEND_BF16=1 OMP_NUM_THREADS=8 GOMP_CPU_AFFINITY="0,2,4,6,8,10,12,14" ./usr/bin/llamafile -m Mistral-Nemo-Instruct-2407.BF16.gguf -c 128 -n 16 -t 0 -p "[INST]bonjour a tu un nom. je ne sais pas comment t'appeler. Si tu n'en as pas je peux t'appeler TINTIN[/INST]"

GGML_USE_BACKEND_BF16=1 OMP_NUM_THREADS=8 GOMP_CPU_AFFINITY="0,2,4,6,8,10,12,14" ./usr/bin/llamafile-bench -m Mistral-Nemo-Instruct-2407.BF16.gguf -n 16 -p "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128,256,512" -r 3
GGML_USE_BACKEND_BF16=1 OMP_NUM_THREADS=8 ./usr/bin/llamafile-bench -m Mistral-Nemo-Instruct-2407.BF16.gguf -n 16 -p "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128,256,512" -r 3
# celui de reference (llamafile)
./usr/bin/llamafile-bench -m Mistral-Nemo-Instruct-2407.BF16.gguf -n 16 -p "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128,256,512" -r 3

 */

/*
QQ note:
 voir ggml_backend_buffer_type_i et creer un buffer...
 - get_alloc_size est appelé avec le tenseur pour recuperer la taille necessaire
 - alloc_buffer => appelé avec une taille a allouer. doit etre configuré si besoin pour le GPU!
 un buffer est "utilisé" pour plusieur tenseur apres!  (cuda n'alloue pas tjs le buffer a ce moment mais dans init_tensor ?
 - get_max_size peut permetre de limiter la taille des buffers allouable (?)

> la suite est dans ggml_backend_buffer_i
  - init_tensor
  - set_tensor ...  (a priory offset seulement pour les kv ???)

# Trace/debug: voir GGML_SCHED_DEBUG=1

 */

#include "ggml-bf16.h"

#ifdef __x86_64__
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "json.h" // https://github.com/nlohmann/json?tab=readme-ov-file#specializing-enum-conversion

#include "ggml-x86_64-immintrin.h"

#include <cstdlib>
#include <cstddef>  // #include <stddef.h>
#include <iostream>
#include <cosmo.h>
#include <string>
#include <regex>
#include <list>

//------------------------------------------------------------------------------------
// #define DO_TIMING
#ifdef DO_TIMING
#include <ctime>
struct mesure {
    struct timespec ts_0;
    void start() {
        clock_gettime(CLOCK_REALTIME, &ts_0);
    }
    double end() {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        auto _s  = ts.tv_sec - ts_0.tv_sec;
        auto _ns = ts.tv_nsec - ts_0.tv_nsec;
        return ((double) _s) + ((double) _ns)/1.e9;
    }
};
#endif // DO_TIMING

// Tools
// - se deplacer dans un buffer en nombre de byte
static inline void* add_byte(void* addr, std::size_t nb) {
    return (void*) (((char*)addr)+nb);
}

// Logs simple sur un flux c++
static inline std::ostream& operator<<(std::ostream& os, enum ggml_type type) {
    return os << ggml_type_name(type);
}

static inline std::ostream& operator<<(std::ostream& os, enum ggml_op type) {
    return os << ggml_op_name(type);
}

// log du tenseur destination
static inline std::ostream& operator<<(std::ostream& os, const struct ggml_tensor * t) {
    // TODO voir a afficher si c'est un tenseur/OP
    os <<"["<<t->ne[0];
    for (int i=1; i<GGML_MAX_DIMS ; i++) {
        os <<":"<<t->ne[i];
    }
    os <<"/"<<t->nb[0];
    for (int i=1; i<GGML_MAX_DIMS ; i++) {
        os <<":"<<t->nb[i];
    }
    return os <<"]@"<< t->type;
}

// log des tenseurs sources
struct log_srcs {
    log_srcs(const ggml_tensor * t): _t(t){}
    const ggml_tensor * _t;
};
static inline std::ostream& operator<<(std::ostream& os, const struct log_srcs t0) {
    auto t = t0._t;
    if (t->src[0]) os<<"s0("<<t->src[0]->name<< (t->src[0]->extra==nullptr?"-":"+") <<")"<<t->src[0];
    for(int i=1; i<GGML_MAX_SRC; i++) {
        if (t->src[i]) os<<", s"<<i<<"("<<t->src[i]->name<<")"<<t->src[i];
    }
    return os;
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

namespace ggml::bf16 {
    enum class TYPE {
        // Tags
        A_CONVERTIR,
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

    class type_from_patern {
        struct elt {
            const ::std::regex name;
            const TYPE t;
        };
        ::std::list<elt> tab;
    public:
        void clear() {
            tab.clear();
        }
        void add(TYPE t, std::string patern){
            //if (t==TYPE::INVALID)
            GGML_ASSERT(t!=TYPE::INVALID); // p'etre etre plus gentil!
            tab.push_back(elt{::std::regex(patern), t});
        }
        TYPE operator()(std::string name){ // const struct ggml_tensor * t ?
            for (auto& e: tab) {
                if (std::regex_match(name, e.name)) {
                    return e.t;
                }
            }
            return TYPE::BF16; // quel valeur par defaut? INVALID?
        }
    };

    // permet de recupere le type du senseur.
    static type_from_patern get_type;

    // gestion des types...
    // TODO ajouter les type composé...
    template<TYPE T> struct type { using t = void; };
#define TYPE_OF(T) ggml::bf16::type<ggml::bf16::TYPE::T>::t
    // le mapping:
    template<> struct type<::ggml::bf16::TYPE::FP32> { using t = float; };

    template<> struct type<::ggml::bf16::TYPE::BF16>      { using t = ::ggml_bf16_t; };
    template<> struct type<::ggml::bf16::TYPE::BF16_32x1> { using t = ::ggml_bf16_t; };
    template<> struct type<::ggml::bf16::TYPE::BF16_8x8>  { using t = ::ggml_bf16_t; };
    template<> struct type<::ggml::bf16::TYPE::BF16_2x16> { using t = ::ggml_bf16_t; };

    template<> struct type<::ggml::bf16::TYPE::FP8>  { using t = ::ggml_fp8_t; };

    //


    class tensor {
    public:
        const TYPE type;
        // float scale = 1;
        tensor(TYPE t) : type(t) {}
        virtual ~tensor(){};
        // TODO:
        // - voir les methodes d'accé au données.
        // voir quel methodes pour gerer les "possibles"
        // - le "choix" de type est static
        // - les tenseur dynamique sont static
        // - les tenseur constant sont instancé/calculé

        static void set(struct ggml_tensor *op, tensor* backend) {
            op->bf16_tensor = backend;
        }
        static tensor* get(const struct ggml_tensor *op) {
            return (tensor*) op->bf16_tensor;
        }
    };

    // TODO: les tenseurs specifique a utiliser...
    // 2 cas:
    //  - les tenseurs complet
    //  - les blocs seuls !!!
    template<typename TYPE, size_t K0, size_t M0, size_t K1, size_t M1>
    class bloc /*: public tensor*/ {
    public:
        bloc() {
            static_assert(K0*M0*sizeof(TYPE) == 512/8);
            // m_values = nullptr;
        }
        // acces element simple.
        TYPE& operator()(size_t i, size_t j) {
            auto i0 = i%K0;
            auto i1 = i/K0;
            auto j0 = j%M0;
            auto j1 = j/M0;
            //return m_values[i0+K0*j0+i1*M0*K0+j1*K0*M0*K1];
            return m_values[j1][i1][j0][i0];
        }
        // acces vecteur.

    private:
        //TYPE* m_values; // stockage [M1][K1][M0][K0] !
        TYPE m_values[M1][K1][M0][K0]; //
    };

    // fait comme ca ca impose des (grosse) containte sur K & M
    template<typename TYPE, size_t M0, size_t N0, size_t M1, size_t N1>
    class tensor_bloc {
    public:
        using bloc_t = bloc<TYPE,M0,N0,M1,N1>;

        // TODO: voir quel autre constructeur sont possible/utils!
        tensor_bloc(TYPE* data, size_t M, size_t N, size_t ld):
            M2(M/(M0*M1)),
            N2(N/(M0*M1))
        {
            GGML_ASSERT(M%(M0*M1) == 0);
            GGML_ASSERT(N%(N0*N1) == 0);
            m_values = new bloc_t[M2*N2];
            // il faut recopier data au bon endroit ...
            for (size_t i=0; i<M; i++) {
                for (size_t j=0; j<M; j++) {
                    (*this)(i,j) = data[i,ld*j];
                }
            }
        }

        TYPE& operator()(size_t i, size_t j) {
            const auto i2 = i/(M0*M1);
            const auto j2 = j/(N0*N1);
            const auto ib = i%(M0*M1);
            const auto jb = j%(N0*N1);
            //
            return m_values[i2+j2*M2](ib,jb);
        }

    private:
        const size_t M2;
        const size_t N2;
        bloc_t* m_values;
    };


    // TODO: gere TYPE + TAILLE_BLOC!
    static tensor tensor_non_supporte_t(TYPE::NON_SUPPORTE);
    static tensor tensor_a_convertir(TYPE::A_CONVERTIR);  // ??? pas forcement utils?
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
template<typename T>
class Matrice {
public:
    inline auto DIM1() const { return m_m; }
    inline auto DIM2() const { return m_n; }
    inline auto LD() const { return m_l; }

    const std::size_t m_m;  // m contigue
    const std::size_t m_n;
    const std::size_t m_l;  // nb elements pour passer a la colonne suivante.
    T* m_values;  // suivant le format ca peut-etre une copie!

    inline Matrice(T* v, std::size_t n, std::size_t m, std::size_t l): m_n(n), m_m(m), m_l(l), m_values(v) {}

    inline Matrice(struct ggml_tensor * t): m_m(t->ne[0]), m_n(t->ne[1]), m_l(t->nb[1]/t->nb[0]), m_values((T*)(t->data)) {
        GGML_ASSERT(t->ne[2]==1);
        GGML_ASSERT(t->ne[3]==1);
    }

    inline Matrice(struct ggml_tensor * t, std::size_t i, std::size_t j): m_m(t->ne[0]), m_n(t->ne[1]), m_l(t->nb[1]/t->nb[0]) { //, m_values((T*)add_byte(t->data,)) {
        m_values = (T*) add_byte(t->data, i*t->nb[2]+j*t->nb[3]);
    }

    inline T& operator()(size_t i, size_t j) {
        return m_values[j*m_l+i];
    }
    inline const T operator()(size_t i, size_t j) const {
        return m_values[j*m_l+i];
    }

    inline T* addr(size_t i, size_t j) {
        return m_values+j*m_l+i;
    }
    inline const T* addr(size_t i, size_t j) const {
        return m_values+j*m_l+i;
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
// - fp32:
static inline auto load(const fp32_t *X) {
    auto x1 = _mm512_loadu_ps(X);
    auto x2 = _mm512_loadu_ps(X+16);
    return _mm512_cvtne2ps_pbh(x2,x1);
}

// - fp16:
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

// write C after last reduction
template<typename... T>
static inline void store(fp32_t *pX, T&&... x) {
    constexpr __mmask16 _m = ((1<<sizeof...(T))-1);
    auto pack = hadd(std::forward<T>(x)...);
    _mm512_mask_storeu_ps(pX, _m, pack);
}

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

class ggml_bf16_op_matmul : public ggml::bf16::op {
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
        //printf("%g us %s %g gigaflops\n", (dt/ITERATIONS)*1000000, #x, (1e-9*2*m*n*k*ITERATIONS)/dt);
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
        auto a = ggml::bf16::tensor::get(A);
        if (!a) return false; // voir si/quand ca peu arriver
        // on sais deja:
        if (a->type == ggml::bf16::TYPE::BF16_32x1) return true;
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

//----------------------------------------------
// cas block de BF16:
namespace ggml::bf16::op_matmul {
    class bf16_2x16: public ggml::bf16::op {

        void exec(struct ggml_tensor *op) const override {
            // TODO
            auto a = ggml::bf16::tensor::get(op->src[0]);
            if (a->type != ggml::bf16::TYPE::BF16_2x16) {
                // => conversion ??? ou doit daja l'etre!!!
            }
        }

        // voir si on fait plusieurs cas 16xN  32xN...
        //bool B_is_allowed(const struct ggml_tensor *B) override {
        //    if (B->ne[1]>6) return false;
        //    return ggml_bf16_op_matmul::B_is_allowed(B);
        //}
        bool A_is_allowed(const struct ggml_tensor *A) override {
            auto a = ggml::bf16::tensor::get(A);
            if (!a) return false; // voir si/quand ca peu arriver
            // on sais deja:
            if (a->type == ggml::bf16::TYPE::BF16_2x16) return true;
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
        template<size_t M1, size_t N0, typename TA, typename TB, typename TC>
        static void gemm(const TA *pA, const TB *pB, TC *pC, std::size_t lda, std::size_t ldb, std::size_t ldc, std::size_t K2) {
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

#pragma GCC unroll N0
            for(size_t j=0; j<N0; j++) {
#pragma GCC unroll (M1/M0)
                for(size_t i=0; i<(M1/M0); i++) {
                    C[i][j] = _mm512_setzero_ps();
                }
            }

            for (size_t k2=0; k2<K2; k2+=K1) {
                // chargement de A
                for (size_t k1=0; k1<K1/K0; ++k1) {
                    for (size_t i1 = 0; i1<M1/M0; ++i1) {
                        //A[k1][i1] = load(pA+M1*K0*k1 + i1*M0*K0); // non sinon pas le meme format que bf16_2x16 ! (bf16_2x32...)
                        A[k1][i1] = load(pA+M1*K0*k1 + i1*M0*lda); // ??? K2 = lda (ou K2*M0 ?)
                        //A[k1][i1] = pA.load_32(M1*K0*k1, i1*M0);  // mieux ? mais faire un control tout n'est pas possible
                        //A[k1][i1] = pA.load_32(M1*k1, i1);
                    }
                }
                for (size_t j=0; j<N0; ++j) {
                    // on charge K1 valeur de B
                    __m128bh B = _mm256_cvtneps_pbh(_mm256_loadu_ps(pB+j*ldb+k2));

                    for (size_t k1=0; k1<K1/K0; ++k1) {
                        auto _B = _mm512_broadcastd_2pbh(B);
                        B = _mm_shiftl_2pbh(B);
                        for (size_t i1=0; i1<M1/M0; ++i1) {
                            C[i1][j] = madd(A[k1][i1], _B, C[i1][j]);
                        }
                    }
                }
            }

            // ecriture de C...
#pragma GCC unroll N0
            for(size_t j=0; j<N0; j++) {
#pragma GCC unroll (M1/M0)
                for (size_t i1=0; i1<M1/M0; ++i1) {
                    store(pC+i1*M0+j*ldc, C[i1][j]);
                }
            }
        }
    };

}


class ggml_bf16_op_matmul_2 : public ggml_bf16_op_matmul {
public:
    // la derniere chance!
    bool A_is_allowed(const struct ggml_tensor *A) override {
        auto a = ggml::bf16::tensor::get(A);
        if (!a) return false; // voir si/quand ca peu arriver
        // on sais deja:
        if (a->type == ggml::bf16::TYPE::BF16_32x1) return true;
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


////////////////////////////////////////
// l'init du backend: [TODO: @ metre sous namespace!]

//------------------------------------------------------------------------------------
// le context definissant le backend
// - ??? comment on gere les tenseurs du backend?
namespace ggml::backend::bf16::tensor {
}
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
        inline ggml::bf16::tensor* get_tensor_type(ggml_tensor * tensor) {
            if (ggml_is_transposed(tensor)) return nullptr;
            ggml::bf16::tensor* t = nullptr;
            if (tensor->buffer && (tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS)) {
                // TODO: les poids ... peuvent supporter plus de transformations que d'autre...
            }
            switch (tensor->type) {
            case GGML_TYPE_BF16 :
                // - est-ce un poid?
                // if (tensor->buffer && (tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS)) {
                //   pour les poids on se permet de convertir / packer le tenseur
                //   la transformation ser faite par le set_tensor
                //   ceux qui sont eligible sont restreint a ceux qui servent au matmul!
                //ggml::bf16::tensor::set(tensor, &ggml::bf16::tensor_bf16_t);
                if (tensor->ne[0] % 32 == 0) {
                    t=&ggml::bf16::tensor_bf16_32x1_t;
                }
            /* TODO: les autres cas possible && "activé"
            if (a->type == ggml::bf16::TYPE::NON_SUPPORTE) return false;
            if (a->type == ggml::bf16::TYPE::BF16
                    && A->ne[0] %  2 == 0  // K%2 == 0
                    && A->ne[1] % 16 == 0  // M%16 == 0
            )
            {
                // OK on va utiliser des bloc de A[16,2] (K0=2, M0=16)
                ggml::bf16::tensor::set(A, &ggml::bf16::tensor_bf16_2x16_t);
                return true;
            }
            */
                break;
            case GGML_TYPE_F32 :
                //  il faut verifier en fonction de type de produit le format a utiliser...
                t=&ggml::bf16::tensor_fp32_t;
                break;
            }
            return t;
        }
        // l'interface: ggml_backend_buffer_i
        inline const char * get_name(void){ return "BF16"; }
        inline void *       get_base(void) { return m_data; }
        inline void         init_tensor(ggml_tensor * tensor) {
            // on configurer tous les tenseur tel que l'on veux qu'il soit:
            auto t = ggml::bf16::tensor::get(tensor);
            if (t == nullptr) {
                // pas encore vu => on va le configurer:
                t = get_tensor_type(tensor);
                // OK on fixe le type !
                if (t) {
                    ggml::bf16::tensor::set(tensor, t);
                } else {
                    ggml::bf16::tensor::set(tensor, &ggml::bf16::tensor_non_supporte_t);
                }
            } else {
                std::cout << "re-init_tensor:" <<tensor->op<< tensor << std::endl;
            }
        }
        inline void         set_tensor(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
            // TODO: @ revoir ca va dependre du type dans le backend!!!
            //  et normalement on sais deja en fct des 2 types ce qu'il y a a faire..
            if (tensor->buffer && (tensor->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS)) {
                //std::cout << "set_tensor(non-weight):" <<tensor->op<< tensor << std::endl;
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
            // TODO: gerer suivant les cas:
            //  - poids pour les matmul => en fct du "type" cible
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
#if 0
        return ggml_backend_cpu_buffer_type();
#endif
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
                    ggml::bf16::op::get(node)->exec(node);
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
                for (auto imp : ggml::bf16::matmul_ops) {
                    // l'ordre est important...
                    if (imp->C_is_allowed(op) &&
                            imp->B_is_allowed(op->src[1]) &&
                            imp->A_is_allowed(op->src[0]))
                    {
                        auto bf16_op = imp->inst(op);
                        if (bf16_op) {
                            ggml::bf16::op::set(op, bf16_op);
                            return true;
                        }
                    }
                }
                return false;
            } else if (op->op == GGML_OP_NONE) {
                // OK deplacé dans l'init, ca devrait etre bon
                if (! ggml::bf16::tensor::get(op)) {
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

        ggml::bf16::matmul_ops.clear();
        // bloc de BF16
        // TODO: if (backend_config == "bf16_2x16") ggml::bf16::matmul_ops.push_back(new ggml::bf16::op_matmul::bf16_2x16);
        // les JARTs en dernier
        ggml::bf16::matmul_ops.push_back(new ggml_bf16_op_matmul_1);
        ggml::bf16::matmul_ops.push_back(new ggml_bf16_op_matmul_2);
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
