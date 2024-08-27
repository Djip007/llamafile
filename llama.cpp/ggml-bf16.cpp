/*
> build:
make clean
make -j16
make install PREFIX=/home/philou/LLM/usr/

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

#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <cosmo.h>
#include <immintrin.h>

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
  unsigned short u=0x8000; // +0?
};

using fp32_t = float;
using bf16_t = struct _bf16_t;

template<typename T> inline bool is(const struct ggml_tensor * t) {return false;}
template<> inline bool is<fp32_t>(const struct ggml_tensor * t) {return t->type==GGML_TYPE_F32;}
template<> inline bool is<bf16_t>(const struct ggml_tensor * t) {return t->type==GGML_TYPE_BF16;}

// gestion de Matrice (en attandant mieux)
template<typename T>
class Matrice {
public:
    inline auto DIM1() const { return m_m; }
    inline auto DIM2() const { return m_n; }
    inline auto LD() const { return m_l; }

    const std::size_t m_m;  // m contigue
    const std::size_t m_n;
    const std::size_t m_l;  // nb elements pour passer a la colonne suivante.
    T* m_values;

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

template<bool RUN> struct type {};
template<> struct type<false>{
    using R = bool;
    using T = const struct ggml_tensor *;
};
template<> struct type<true>{
    using R = void;
    using T = struct ggml_tensor *;
};

//------------------------------------------------------------------------------------
// le context definissant le backend
class ggml_backend_bf16_context {
public:
    // le nom du backend
    inline const char * name() { return "BF16"; }

    // le type de buffer gere par defaut sur ce backend
    // TODO: creer nos propres buffer avec (ggml_backend_buffer_type_i.is_host => true)
    inline ggml_backend_buffer_type_t get_default_buffer_type() { return ggml_backend_cpu_buffer_type(); }

    // execution du graph contenant les OPs "supported"
    inline enum ggml_status graph_compute(struct ggml_cgraph * cgraph) {
        std::cout << "." ;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            struct ggml_tensor * node = cgraph->nodes[i];

            switch (node->op) {
                case GGML_OP_MUL_MAT:
                    // std::cout << " > " <<node->op<<"("<<log_srcs(node)<<" => "<<node<<"): "<< node->name<<std::endl;
                    mul_mat<true>(node);
                    break;

                //case GGML_OP_OUT_PROD:
                //    ggml_backend_blas_out_prod(ctx, node);
                //    break;

                // y a ca dans backend OPENBLAS ... sais pas pourquoi.
                case GGML_OP_NONE:
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
        // dispach des OPs
        if (op->op == GGML_OP_MUL_MAT) {
            std::cout << "TRACE> " <<op->op<<"@"<<op->name<<" ("<<log_srcs(op)<<" => "<<op<<")"<<std::endl;
            if (mul_mat<false>(op)) return true;
            //std::cout << "TODO> " <<op->op<<"@"<<op->name<<" ("<<log_srcs(op)<<" => "<<op<<")"<<std::endl;
        //} else if (op->op == GGML_OP_OUT_PROD) {
        //    std::cout << " > " <<op->op<<"("<<log_srcs(op)<<" => "<<op<<"): "<< op->name<<std::endl;
        }
        //std::cout << " > " <<op->op<<"("<<log_srcs(op)<<" => "<<op<<"): "<< op->name<<std::endl;
        return false;
    }

    // ???
    inline bool supports_buft(ggml_backend_buffer_type_t buft) {
        return ggml_backend_buft_is_host(buft);
    }

private:

    // GGML_OP_MUL_MAT.
    template<bool RUN> // peut etre appelé pour le calcul (RUN=true) ou son support (false)
    inline auto mul_mat(typename type<RUN>::T op) -> typename type<RUN>::R {
        // => src0 = poids src1 = In => DST = T(SRC0) * SRC1 ...
        //  > MUL_MAT( [14336:4096]@bf16, s1 [14336:7]@f32 =>  [4096:7]@f32): ffn_out-29
        const auto src0 = op->src[0];
        const auto src1 = op->src[1];
        auto dst  = op;

        // exemple:
        // => control si les types sont supporté
        if( Matrice<bf16_t>::valid(src0) &&
            Matrice<fp32_t>::valid(src1) &&
            Matrice<fp32_t>::valid(op)   &&
            src0->ne[0] % 32 == 0 // K%32==0 => pour l'instant pas d'autre cas... mais c'est (presque) tjs le cas!
            ) {
            if constexpr (RUN) { // execution
#ifdef DO_TIMING
                mesure time; time.start();
#endif
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
            } else {
                return true;
            }
        }
        // autre version possible
        // if(...)
        if constexpr (!RUN) {
            return false;
        }
    }
    // implementation de l'OP pour les differents types
    //  - bf16+fp32=fp32
    void mul_mat(const Matrice<bf16_t>& A, const Matrice<fp32_t>& B, Matrice<fp32_t>& C);
    //  - autres cas ?

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
// => implemetation des OPs

void ggml_backend_bf16_context::mul_mat(const Matrice<bf16_t>& A, const Matrice<fp32_t>& B, Matrice<fp32_t>& C) {
    const auto m = C.DIM1(); // == A.DIM2()
    const auto n = C.DIM2(); // == B.DIM2()
    const auto k = A.DIM1(); // == B.DIM1()
    GGML_ASSERT(A.LD()>=k);
    GGML_ASSERT(B.LD()>=k);
    GGML_ASSERT(C.LD()>=m);
    // TODO: l'implementation.
    //[...]
}
// end #include "ggml-bf16/matmul.cpp"

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
        /* .supports_buft           = */ ggml_backend_bf16_supports_buft,
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
    ggml_backend_t backend = nullptr;
    // TODO: voir sous quel condition activer ca:
    if (X86_HAVE(FMA) && X86_HAVE(F16C) && X86_HAVE(AVX2) && X86_HAVE(AVX512F) && X86_HAVE(AVX512BW) && X86_HAVE(AVX512DQ) && X86_HAVE(AVX512VL) && X86_HAVE(AVX512_BF16)) {
        // voir quel contrainte mettre ... et est-ce que ca doit etre ici?
        ggml_backend_bf16_context * ctx = new ggml_backend_bf16_context;

        backend = new ggml_backend {
            /* .guid      = */ ggml_backend_bf16_guid(),
            /* .interface = */ blas_backend_i, // => bf16_backend_i
            /* .context   = */ ctx,
        };
    }
    return backend;
}
#else
ggml_backend_t ggml_backend_bf16_init(void) {
    return nullptr;
}
#endif
