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
#include "ggml-bf16.h"

#ifdef __x86_64__
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "json.h" // https://github.com/nlohmann/json?tab=readme-ov-file#specializing-enum-conversion

#include <cstdlib>
#include <cstddef>  // #include <stddef.h>
#include <iostream>
#include <cosmo.h>
#include <string>
#include <regex>
#include <list>

#include <immintrin.h>
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

        static void set(const struct ggml_tensor *op, tensor* backend) {
            // normalement a ne pas faire ;)
            const_cast<struct ggml_tensor *>(op)->bf16_op = backend;
        }
        static tensor* get(const struct ggml_tensor *op) {
            // normalement a ne pas faire ;)
            return (tensor*) const_cast<struct ggml_tensor *>(op)->bf16_op;
        }
    };

    // TODO: les tenseurs specifique a utiliser...
    //class tensor_pack : public tensor { };

    static tensor tensor_non_supporte_t(TYPE::NON_SUPPORTE);
    static tensor tensor_a_convertir(TYPE::A_CONVERTIR);
    static tensor tensor_fp32_t(TYPE::FP32);
    static tensor tensor_bf16_32x1_t(TYPE::BF16_32x1);
    static tensor tensor_bf16_t(TYPE::BF16);

    // que des instances "static"
    // Arbre de decision ?
    class op {
    public:
        virtual ~op() {}
        // - is_alowed => TODO...
        virtual bool C_is_allowed(const struct ggml_tensor *C) {
            // TODO @ mettre dans un fils!
            if (C->type != GGML_TYPE_F32) return false;
            if (ggml_is_transposed(C)) return false;
            return true;
        }
        virtual bool B_is_allowed(const struct ggml_tensor *B) {
            // TODO @ mettre dans un fils!
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
// le context definissant le backend
class ggml_backend_bf16_context {
public:

    ~ ggml_backend_bf16_context() {
        // TODO: il faut faire le menage de tout ce que le backend a cree.
        //   si on ne peu pas le netoyé plus finement:
        // - store all tensor (ggml_bf16_op)
        // - free all of them!
    }

    // le nom du backend
    inline const char * name() { return "BF16"; }

    // le type de buffer gere par defaut sur ce backend
    // TODO: creer nos propres buffer avec (ggml_backend_buffer_type_i.is_host => true)
    inline ggml_backend_buffer_type_t get_default_buffer_type() { return ggml_backend_cpu_buffer_type(); }

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
        //  - definir ce qui va servir pour le calcul
        //  - comme ce n'est pas conservé => instance "static" fct de la config
        //      =>  plus rapide pour le calcul il sais ce qu'il a a faire!
        // si A n'est pas en init on sais plus vite quoi faire... (il peu porter le calcul???)
        // si A est en init il faudra choisir ce que l'on veux faire:
        //    -> le choix de la quantisation a utilisé peur dependre d'une config externe
        //    => avoir une config pour chaque poids "utils" et le type de quantisation...
        //       avec 1 par defaut!!
        if (op->op == GGML_OP_MUL_MAT) {
            // les poids sont soit:
            //  - deja reformaté
            //  - en init.
            // les poids qui sont des caches:
            //   - deja tagé ...
            // les A qui sont des op:
            //  - sans config ...

            // TODO: choisir la 1er OP possible ???
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
            // des tenseurs simple: poids ou
            auto t = ggml::bf16::tensor::get(op);
            if (t == nullptr) {
                // pas encore vu => on va le configurer:
                // - est-ce un poid?
                if (op->buffer && (op->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS)) {
                    // oui => on peu le convertir: (pour ceux que l'on sais faire..
                    // ggml::bf16::tensor::set(op, &ggml::bf16::tensor_a_convertir);
                    switch (op->type) {
                    case GGML_TYPE_BF16 :
                        ggml::bf16::tensor::set(op, &ggml::bf16::tensor_bf16_t);
                        break;
                    default:
                        ggml::bf16::tensor::set(op, &ggml::bf16::tensor_non_supporte_t);
                    }
                } else {
                    // @ priori un cache => pas de conversion "possible"
                    switch (op->type) {
                    case GGML_TYPE_BF16 :
                        ggml::bf16::tensor::set(op, &ggml::bf16::tensor_bf16_t);
                        break;
                    default:
                        ggml::bf16::tensor::set(op, &ggml::bf16::tensor_non_supporte_t);
                    }
                }
                t = ggml::bf16::tensor::get(op);
            }
            return t->type != ggml::bf16::TYPE::NON_SUPPORTE;
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

//------------------------------------------------------------------------------------
// les methodes du backend => wrapper (elle sont codé dans la classe)
GGML_CALL static const char * ggml_backend_bf16_name(ggml_backend_t backend) {
    ggml_backend_bf16_context * ctx = (ggml_backend_bf16_context *)backend->context;
    return ctx->name();
}

GGML_CALL static void ggml_backend_bf16_free(ggml_backend_t backend) {
    ggml_backend_bf16_context * ctx = (ggml_backend_bf16_context *)backend->context;
    delete ctx;
    delete backend;
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_bf16_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_bf16_context * ctx = (ggml_backend_bf16_context *)backend->context;
    return ctx->get_default_buffer_type();
}

GGML_CALL static enum ggml_status ggml_backend_bf16_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_bf16_context * ctx = (ggml_backend_bf16_context *)backend->context;
    return ctx->graph_compute(cgraph);
}

GGML_CALL static bool ggml_backend_bf16_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    ggml_backend_bf16_context * ctx = (ggml_backend_bf16_context *)backend->context;
    return ctx->supports_op(op);
}

GGML_CALL static bool ggml_backend_bf16_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    ggml_backend_bf16_context * ctx = (ggml_backend_bf16_context *)backend->context;
    return ctx->supports_buft(buft);
}

//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------
// => implemetation des OPs

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
        if (B->ne[1]>6) return false;
        return ggml_bf16_op_matmul::B_is_allowed(B);
    }
    bool A_is_allowed(const struct ggml_tensor *A) override {
        auto a = ggml::bf16::tensor::get(A);
        if (!a) return false; // voir si/quand ca peu arriver
        // on sais deja:
        if (a->type == ggml::bf16::TYPE::BF16_32x1) return true;
        if (a->type == ggml::bf16::TYPE::NON_SUPPORTE) return false;
        if (ggml_is_transposed(A)) return false;
        if (a->type == ggml::bf16::TYPE::BF16 && A->ne[0] % 32 == 0) {
            ggml::bf16::tensor::set(A, &ggml::bf16::tensor_bf16_32x1_t);
            return true;
        }
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
        constexpr size_t M0 = 4; //6;
        constexpr size_t N0 = 6; //;
        constexpr size_t M1 = 10;
        constexpr size_t K0 = 4096;
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
        auto a = ggml::bf16::tensor::get(A);
        if (!a) return false; // voir si/quand ca peu arriver
        // on sais deja:
        if (a->type == ggml::bf16::TYPE::BF16_32x1) return true;
        if (a->type == ggml::bf16::TYPE::NON_SUPPORTE) return false;
        if (ggml_is_transposed(A)) return false;
        if (a->type == ggml::bf16::TYPE::BF16 && A->ne[0] % 32 == 0) {
            ggml::bf16::tensor::set(A, &ggml::bf16::tensor_bf16_32x1_t);
            return true;
        }
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
        constexpr size_t M1 = 10;
        constexpr size_t N1 = 4;
        constexpr size_t K0 = 4096;
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
// l'init du backend
static struct ggml_backend_i blas_backend_i = {
        /* .get_name                = */ ggml_backend_bf16_name,  // wraper(ggml_backend_bf16_context::name)
        /* .free                    = */ ggml_backend_bf16_free,
        /* .get_default_buffer_type = */ ggml_backend_bf16_get_default_buffer_type,
        /* .set_tensor_async        = */ nullptr,  // voir ggml_backend_tensor_set_async
        /* .get_tensor_async        = */ nullptr,
        /* .cpy_tensor_async        = */ nullptr,
        /* .synchronize             = */ nullptr,
        /* .graph_plan_create       = */ nullptr,
        /* .graph_plan_free         = */ nullptr,
        /* .graph_plan_update       = */ nullptr,
        /* .graph_plan_compute      = */ nullptr,
        /* .graph_compute           = */ ggml_backend_bf16_graph_compute,
        /* .supports_op             = */ ggml_backend_bf16_supports_op,
        /* .supports_buft           = */ ggml_backend_bf16_supports_buft,  // si tout les entrée sont suporté l'op devrai lui etre assigné??? enfin pas sur!!
        /* .offload_op              = */ nullptr,
        /* .event_new               = */ nullptr,
        /* .event_free              = */ nullptr,
        /* .event_record            = */ nullptr,
        /* .event_wait              = */ nullptr,
        /* .event_synchronize       = */ nullptr,
};

static ggml_guid_t ggml_backend_bf16_guid(void) {
    static ggml_guid guid = { 0xca, 0xfe, 0xde, 0xca, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01 };
    return &guid;
}

ggml_backend_t ggml_backend_bf16_init(void) {
    if (getenv("GGML_USE_BACKEND_BF16") == nullptr) return nullptr;
    std::string backend_config{getenv("GGML_USE_BACKEND_BF16")};
    // changer la config en fonction de l'option choisi:
    // backend_config == "bf16_32x1"
    // backend_config == "E4M3_8x8_DAZ" // (denormals-are-zero)"
    // backend_config == "E4M3_8x8"     // process subnormal FP8 nomber on BF16 conversion
    // ...
    ggml_backend_t backend = nullptr;
    // TODO: voir sous quel condition activer ca:
    if (X86_HAVE(FMA) && X86_HAVE(F16C) && X86_HAVE(AVX2) && X86_HAVE(AVX512F) && X86_HAVE(AVX512BW) && X86_HAVE(AVX512DQ) && X86_HAVE(AVX512VL) && X86_HAVE(AVX512_BF16)) {
        // voir quel contrainte mettre ... et est-ce que ca doit etre ici?
        ggml_backend_bf16_context * ctx = new ggml_backend_bf16_context;

        backend = new ggml_backend {
            /* .guid      = */ ggml_backend_bf16_guid(/*config?*/),
            /* .interface = */ blas_backend_i, // => bf16_backend_i
            /* .context   = */ ctx,
        };

        //if (ggml::bf16::op_jart_1 == nullptr) ggml::bf16::op_jart_1 = new ggml_bf16_op_matmul_1;
        //if (ggml::bf16::op_jart_2 == nullptr) ggml::bf16::op_jart_2 = new ggml_bf16_op_matmul_2;
        // il faut les mettre dans l'ordre "priviligé"!
        ggml::bf16::matmul_ops.clear();
        ggml::bf16::matmul_ops.push_back(new ggml_bf16_op_matmul_1);
        ggml::bf16::matmul_ops.push_back(new ggml_bf16_op_matmul_2);
    }
    return backend;
}
#else
ggml_backend_t ggml_backend_bf16_init(void) {
    return nullptr;
}
#endif
