//#define __x86_64__
/*
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
export RUN="./usr/bin/llamafile -m Mistral-7B-Instruct-v0.3.BF16.gguf   -c 128 -n 16 -t 0 -s 42 -p "
export RUN="./usr/bin/llamafile -m Mistral-Nemo-Instruct-2407.BF16.gguf -c 128 -n 16 -t 0 -s 42 -p "
export RUN_ARGS="[INST]bonjour a tu un nom. je ne sais pas comment t'appeler. Si tu n'en as pas je peux t'appeler TINTIN[/INST]"

> les [jart]
OMP_NUM_THREADS=8  GGML_USE_BACKEND_BF16='["NONE     "]'                           $RUN "${RUN_ARGS}"
OMP_NUM_THREADS=8  GGML_USE_BACKEND_BF16='["BF16_32x1_5x5"]'                       $RUN "${RUN_ARGS}"
OMP_NUM_THREADS=8  GGML_USE_BACKEND_BF16='["BF16_32x1_4x6"]'                       $RUN "${RUN_ARGS}"
OMP_NUM_THREADS=8  GGML_USE_BACKEND_BF16='["FP8_E4M3C_32x1_5x5", "BF16_32x1_5x5"]' $RUN "${RUN_ARGS}"
OMP_NUM_THREADS=8  GGML_USE_BACKEND_BF16='["FP8_E4M3C_32x1_4x6", "BF16_32x1_4x6"]' $RUN "${RUN_ARGS}"

> les miens
OMP_NUM_THREADS=8  GGML_USE_BACKEND_BF16='["BF16_2x16",      "BF16_32x1_5x5"]'  $RUN "${RUN_ARGS}"
OMP_NUM_THREADS=8  GGML_USE_BACKEND_BF16='["FP8_E4M3G_2x16", "BF16_32x1_5x5"]'  $RUN "${RUN_ARGS}"
# OMP_NUM_THREADS=8  GGML_USE_BACKEND_BF16='["FP8_E4M3C_2x16", "BF16_32x1_5x5"]'  $RUN "${RUN_ARGS}"

#>  pas toujours terrible avec un scale global...
OMP_NUM_THREADS=8  GGML_USE_BACKEND_BF16='["FP8_E4M3G_32x1_5x5", "BF16_32x1_5x5"]' $RUN "${RUN_ARGS}"
OMP_NUM_THREADS=8  GGML_USE_BACKEND_BF16='["FP8_E4M3G_32x1_4x6", "BF16_32x1_4x6"]' $RUN "${RUN_ARGS}"
OMP_NUM_THREADS=8  GGML_USE_BACKEND_BF16='["FP8_E4M3G_2x16",     "BF16_32x1_5x5"]' $RUN "${RUN_ARGS}"


# GGML_USE_BACKEND_BF16=1 OMP_NUM_THREADS=8 GOMP_CPU_AFFINITY="0,2,4,6,8,10,12,14" ./usr/bin/llamafile -m Mistral-Nemo-Instruct-2407.BF16.gguf -c 128 -n 16 -t 0 -p "[INST]bonjour a tu un nom. je ne sais pas comment t'appeler. Si tu n'en as pas je peux t'appeler TINTIN[/INST]"
# GGML_USE_BACKEND_BF16=1 OMP_NUM_THREADS=8 GOMP_CPU_AFFINITY="0,2,4,6,8,10,12,14" ./usr/bin/llamafile-bench -m Mistral-Nemo-Instruct-2407.BF16.gguf -n 16 -p "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128,256,512" -r 3
# celui de reference (llamafile)

> benchmarks:
export RUN="./usr/bin/llamafile-bench -m Mistral-Nemo-Instruct-2407.BF16.gguf -n 16 -r 3 -p "
export RUN_ARGS="1,1,1,2,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128,256,512"


GGML_USE_BACKEND_BF16='["FP8_E4M3_32x1", "BF16_32x1"]' OMP_NUM_THREADS=8  \
./usr/bin/llamafile-bench -m Mistral-Nemo-Instruct-2407.BF16.gguf -n 16 -p "1,1,1,2,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128,256,512" -r 3

./usr/bin/llamafile-bench -m Mistral-Nemo-Instruct-2407.BF16.gguf -n 16 -p "1,1,1,2,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128,256,512" -r 3


#> mesure de perplexité:
// [jart]: 11.7748 vs. 5.6997 ?  (wiki.test.raw?)
wget https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1.zip
=> wikitext-2-raw/wiki.valid.raw

GGML_USE_BACKEND_BF16='["FP8_E4M3G_32x1", "BF16_32x1"]' OMP_NUM_THREADS=8 \
GGML_USE_BACKEND_BF16='["FP8_E4M3C_32x1", "BF16_32x1"]' OMP_NUM_THREADS=8 \
GGML_USE_BACKEND_BF16='["FP8_E4M3G_2x16",  "BF16_32x1_5x5"]' OMP_NUM_THREADS=8 \
GGML_USE_BACKEND_BF16='["FP8_E4M3C_32x1_4x6", "BF16_32x1_4x6"]' OMP_NUM_THREADS=8 \
./usr/bin/llamafile-perplexity -m Mistral-Nemo-Instruct-2407.BF16.gguf -f wikitext-2-raw/wiki.valid.raw -s 31337

 */

// # Trace/debug: voir GGML_SCHED_DEBUG=1
//     LLAMA_LOG_INFO("%s: model configured\n", __func__);

/*
TODO:
 - revoir la facon de configurer les threads OpenMP...
 - voir comment gerer les type utilisable.

 */

#include "ggml-bf16.h"

#ifdef __x86_64__
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "json.h" // https://github.com/nlohmann/json?tab=readme-ov-file#specializing-enum-conversion
using json = nlohmann::json;

#include "ggml-x86_64-immintrin.h"

#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <cosmo.h>
#include <string>
#include <regex>
#include <list>

static constexpr std::size_t TENSOR_ALIGNMENT = 64; // le meme que pour les buffer!

// qq includes pour pas avoir tout dans le meme fichier...
#include "ggml-bf16-log.inc"
#include "ggml-bf16-type.inc"
#include "ggml-bf16-matrice.inc"

namespace ggml::backend::bf16 {
    enum class TYPE {
        // Tags
        // les types de base sans block (format d'origine)
        FP32 = GGML_TYPE_F32,
        FP16 = GGML_TYPE_F16,
        BF16 = GGML_TYPE_BF16,
        //FP8  = GGML_TYPE_FP8, // F8_E4M3 ???

        // les types non d'origine
        TOUS = GGML_TYPE_COUNT+1,
        NON_SUPPORTE,
        E3M4,
        E4M3,
        E5M2,
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

    // un helper...
    template<typename T> struct type_of {using t=void;};
    template<> struct type_of<fp32_t>    { static constexpr TYPE T=TYPE::FP32; static constexpr ggml_type G=GGML_TYPE_F32;   };
    template<> struct type_of<bf16_t>    { static constexpr TYPE T=TYPE::BF16; static constexpr ggml_type G=GGML_TYPE_BF16;  };
    template<> struct type_of<f8_E3M4_t> { static constexpr TYPE T=TYPE::E3M4; static constexpr ggml_type G=GGML_TYPE_COUNT; };
    template<> struct type_of<f8_E4M3_t> { static constexpr TYPE T=TYPE::E4M3; static constexpr ggml_type G=GGML_TYPE_COUNT; };
    template<> struct type_of<f8_E5M2_t> { static constexpr TYPE T=TYPE::E5M2; static constexpr ggml_type G=GGML_TYPE_COUNT; };

    class Tensor_t {
    public:
        const std::string NAME;
        enum class COMPAT {
            NATIVE,
            CONVERT,
            NONE
        };
    public:
        Tensor_t(const std::string& name): NAME(name){}
        virtual ~Tensor_t() {}

        // les enregistrements:
        static void SET(struct ggml_tensor *op, Tensor_t* backend) {
            op->bf16_tensor = backend;
        }
        static Tensor_t* GET(const struct ggml_tensor *op) {
            return (Tensor_t*) op->bf16_tensor;
        }

        virtual COMPAT is_allowed(const struct ggml_tensor *op) = 0;
        virtual size_t get_alloc_size(const struct ggml_tensor * tensor) = 0;
        virtual void set_tensor(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) = 0;
    };

    class AnyTensor : public Tensor_t {
    public:
        AnyTensor():Tensor_t("TOUS") {};
        COMPAT is_allowed(const struct ggml_tensor *op) override { return COMPAT::NATIVE; }
        size_t get_alloc_size(const struct ggml_tensor * tensor) override { return ggml_nbytes(tensor); }
        virtual void set_tensor(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
            memcpy((char *)tensor->data + offset, data, size);
        }
    };

    // de quoi gerer les tenseurs...
    template<typename T, int K0=1, int M0=1, Scale SCALE=Scale::NONE>
    class tensor : public Tensor_t {
        using type_t = T;
        static constexpr auto _T = type_of<T>::T;
        static constexpr auto _G = type_of<T>::G;
    public:
        // TODO les tailles des blocs!  M0/N0/M1/N1 ou K0/M0/K1/M1
        //      "constant" => WEIGHT => changement de type et repack possible!
        tensor(const std::string name) : Tensor_t(name) {}

        // les "proprietes:
        //  - est-ce qu'il est possible depuis ce type de tenseur
        //  Il faut que la "conversion" existe!
        COMPAT is_allowed(const struct ggml_tensor *op) override {
            // - possible sans pading (voir si on autoriserais pas un peu de pading si ca arrive)
            if (op->ne[0]%K0 != 0) return COMPAT::NONE;
            // - est-ce possible avec les données d'origines.
            if constexpr (M0==1) {
                if (op->type == _G) {
                    // si c'est contigue c'est bon...
                    if (ggml_is_contiguous(op)) return COMPAT::NATIVE;
                }
            }

            // OK est-ce possible avec une conversion:
            //  => seulement si c'est un poids!
            if ( (op->flags & GGML_TENSOR_FLAG_WEIGHTS) == GGML_TENSOR_FLAG_WEIGHTS ) {
                // - possible sans pading
                if (op->ne[1]%M0 != 0) return COMPAT::NONE;
                // - 2D simple (voir comment gerer les autres cas...)
                if (op->ne[2] != 1) return COMPAT::NONE;
                if (op->ne[3] != 1) return COMPAT::NONE;
                // - simple re-order...
                if (op->type == _G) return COMPAT::CONVERT;
                // - conversion de type:
                if constexpr (_T == TYPE::FP32) {
                    switch (op->type) {
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_F16:
                        // un peu bete... et pas codé de toute facon
                        break;
                    }
                } else
                if constexpr (_T == TYPE::BF16) {
                    switch (op->type) {
                    case GGML_TYPE_F32:
                        return COMPAT::CONVERT;
                    case GGML_TYPE_F16:
                        break; // @ voir...
                    }
                } else
                if constexpr ((_T == TYPE::E3M4) || (_T == TYPE::E4M3) || (_T == TYPE::E5M2)) {
                    // depuis FP32/BF16/FP16
                    switch (op->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_BF16:
                        return COMPAT::CONVERT;
                    case GGML_TYPE_F16:
                        break; // TODO:...
                    }
                } else {
                    static_assert(false, "TYPE non supporté");
                }
            }
            return COMPAT::NONE;
        }

        // - quel taille faut-il pour stoquer ces données?
        inline size_t get_alloc_size(const struct ggml_tensor * tensor) {
            // ne doit etre appelé que si on est sur de vouloir l'utiliser.
            if (is_allowed(tensor) == COMPAT::CONVERT) {
                // calcul de sa taille:
                auto size = sizeof(T)*tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
                return size+Matrice<T,K0,M0,SCALE>::scale_size(tensor->ne[1]);
            }
            // OK pour NATIVE ou NONE
            return ggml_nbytes(tensor);
        }

        //  set:
        inline void set_tensor(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
            if constexpr (M0 == 1) {
                if (tensor->type == _G) {
                    // pas de conversion a faire:
                    memcpy((char *)tensor->data + offset, data, size);
                    return;
                }
            }
            // dans tous les autres cas il y a copie...
            //  et je sais pas le faire dans tous les cas..
            GGML_ASSERT(offset == 0);
            if (size != ggml_nbytes(tensor)) GGML_ABORT("BF16_ABORT(%s: %ld != %ld)", __PRETTY_FUNCTION__, size, get_alloc_size(tensor));
            GGML_ASSERT(tensor->ne[2] == 1);
            GGML_ASSERT(tensor->ne[3] == 1);
            // pas le choix pour la destination:
            // Matrice<T,K0,M0> dest(tensor);        // voir quel methodes pour gerer les "possibles"
            // - le "choix" de type est static
            // - les tenseur dynamique sont static
            // - les tenseur constant sont instantié/calculé

            switch (tensor->type) {
            case GGML_TYPE_F32:
                set_tensor<fp32_t>(tensor, data);
                break;
            case GGML_TYPE_BF16:
                set_tensor<bf16_t>(tensor, data);
                break;
            default:
                std::cout << "Type pas gere: " << __PRETTY_FUNCTION__ << tensor->name << tensor << std::endl;
                GGML_ASSERT(tensor->type == (int)TYPE::NON_SUPPORTE);
                //memcpy((char *)tensor->data + offset, data, size);
            }
        }

    private:
        template<typename T2>
        void set_tensor(struct ggml_tensor * tensor, const void * data) {
            Matrice<T,K0,M0,SCALE> dest(tensor);
            const Matrice<const T2> orig(data, tensor);
            float scale = 1;
            if constexpr(SCALE == Scale::GLOBAL) {
                // le facteur global
                auto max = orig.max();
                scale = MAX<T>()/max;
                dest.set_scale((max/MAX<T>())*CORRECTION<T>());
            }
#pragma omp parallel for schedule(guided)
            for (int j=0; j<orig.DIM2(); j++) {
                if constexpr(SCALE==Scale::PER_COL) {
                    // 1 coef par colonne
                    auto max = orig.max(j);
                    scale = MAX<T>()/max;
                    dest.set_scale((max/MAX<T>())*CORRECTION<T>(),j);
                }
                for (int i=0; i<orig.DIM1(); i++) {
                    if constexpr(SCALE==Scale::NONE) {
                        conv(dest(i,j), orig(i,j));
                    } else {
                        float v;
                        conv(v, orig(i,j));
                        conv(dest(i,j), v*scale);
                    }
                }
            }
        }
    };

    // les types de tenseur que l'on sais traiter...
    // TODO: ajouter const ...
    // - les tenseur que l'on ne peu/sais pas gerer.
    static AnyTensor tensor_non_supporte_t;

    // singleton sur les types de template.
    template<typename T, int K, int M=1, Scale SCALE=Scale::NONE>
    static constexpr Tensor_t* tensor_type() {
        static tensor<T,K,M,SCALE> type("tensor<"+std::to_string<T>()+","+std::to_string(K)+","+std::to_string(M)+","+std::to_string(SCALE)+">");
        return &type;
    }

    // la liste que l'on a le droit d'utiliser, dans l'ordre de priorité:
    static std::list<Tensor_t*> tensors;

    // que des instances "static"
    // Arbre de decision ?
    class op {
    public:
        virtual ~op() {}

        // - ? gestion format des poids / choix du tenseur?
        static void SET(const struct ggml_tensor *ggml_op, op* bf16_op) {
            // normalement a ne pas faire ;)
            const_cast<struct ggml_tensor *>(ggml_op)->bf16_op = bf16_op;
        }
        static inline op* GET(struct ggml_tensor *ggml_op) {
            return (op*) ggml_op->bf16_op;
        }

        // pour l'instant tjs des type FP32 natif ... pas le choix et y en a pas d'autre!
        virtual bool C_is_allowed(const struct ggml_tensor *C) {
            return tensor_type<fp32_t,32>()->is_allowed(C) == Tensor_t::COMPAT::NATIVE;
        }
        virtual bool B_is_allowed(const struct ggml_tensor *B) {
            return tensor_type<fp32_t,32>()->is_allowed(B) == Tensor_t::COMPAT::NATIVE;
        }
        virtual bool A_is_allowed(const struct ggml_tensor *A) = 0;
        // - compute
        virtual void exec(struct ggml_tensor *op) const = 0;
    };

    // les ops possibles:
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


//----------------------------------------------
// - gemm comme [jart]:
#include "ggml-bf16-sgemm.inc"

//----------------------------------------------
// cas block de BF16:
namespace ggml::bf16::op_matmul {
    template<typename TYPE_A, Scale SCALE>
    class bf16_2x16: public ggml::backend::bf16::op {
        // static constexpr auto SCALE = ggml::backend::bf16::type_of<TYPE_A>::SCALE;
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
            const Matrice<TYPE_A,2,16,SCALE> A(op->src[0]);
            const Matrice<fp32_t>            B(op->src[1]);
            Matrice<fp32_t>                  C(op);
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
        //    // aussi possible avec ggml::backend::bf16::tensor_fp32_8x1_t
        //    return ggml_bf16_op_matmul::B_is_allowed(B);
        //}
        bool A_is_allowed(const struct ggml_tensor *A) override {
            auto a = ggml::backend::bf16::Tensor_t::GET(A);
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
            if (a == ggml::backend::bf16::tensor_type<TYPE_A,2,16,SCALE>()) return true;
            return false;
        }

        // le bloc de bas niveau
        //   pas le plus performant possible mais de quoi faire qq tests
        //  => K0/M0 sont alors fixé pour A!
        template<size_t N0, typename TA, typename TB, typename TC>
        static void gemm(const TA *pA, const TB *pB, TC *pC, std::size_t lda, std::size_t ldb, std::size_t ldc, std::size_t K2, float scale) {
            // lda: comment passer de A[k,i] => A[k,i+M1]
            // ldb: comment passer de B[k,j] => B[k,j+1]
            // ldc: comment passer de C[i,j] => C[i,j+1]
            constexpr int K0 =  2; //
            constexpr int M0 = 16; // 16 FP32 => 1 AVX512!
            constexpr int K1 =  8; // des blocks de 4 (8/2) pour ameloré les lecture de B!
            static_assert(N0>0);
            GGML_ASSERT(K2%K1 == 0);

            // K%32 == 0!!
            // A[?,K+:lda]
            // B[?,K+:ldb]
            // C[?,ldc]
            __m512   C[N0];    // m512   == fp32[M0]
            __m512bh A[K1/K0]; // m512bh == bf16[M0][K0]

            //std::cout << "  - 0" << std::endl;
#pragma GCC unroll N0
            for(size_t j=0; j<N0; j++) {
                C[j] = _mm512_setzero_ps();
            }

            for (size_t k2=0; k2<K2; k2+=K1) { // de 8 en 8 ...
                // chargement de A
#pragma GCC unroll K1
                for (size_t k1=0; k1<K1/K0; ++k1) {  // [0..3]
                    A[k1] = load(pA + k2*M0 + k1*M0*K0); // lda == K2*M0 ...
                }
#pragma GCC unroll N0
                for (size_t j=0; j<N0; ++j) {  // [0..~16]
                    // on charge K1 valeur de B
                    __m128bh B = _mm256_cvtneps_pbh(_mm256_loadu_ps(pB+j*ldb+k2));
#pragma GCC unroll K1
                    for (size_t k1=0; k1<K1/K0; ++k1) {  // [0..4]
                        auto _B = _mm512_broadcastd_2pbh(B);
                        B = _mm_shiftl_2pbh(B);
                        // C[j] = madd(A[k1], _B, C[j]);
                        C[j] = _mm512_dpbf16_ps(C[j], A[k1], _B);
                    }
                }
            }

            // ecriture de C...
#pragma GCC unroll N0
            for(size_t j=0; j<N0; j++) {
                store(pC+j*ldc, C[j]* (SCALE!=Scale::NONE?scale:1));
            }
        }

        template<size_t N0, size_t M0, size_t K0>
        inline void sgemm_bloc(const Matrice<TYPE_A,K0,M0,SCALE>& A, const Matrice<fp32_t>& B, Matrice<fp32_t>& C, size_t i, size_t j, size_t N, size_t K) const {
            static_assert(N0<=16);
            switch (N) {
                case 16: gemm<16>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K, A.get_scale()); break;
                case 15: gemm<15>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K, A.get_scale()); break;
                case 14: gemm<14>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K, A.get_scale()); break;
                case 13: gemm<13>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K, A.get_scale()); break;
                case 12: gemm<12>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K, A.get_scale()); break;
                case 11: gemm<11>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K, A.get_scale()); break;
                case 10: gemm<10>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K, A.get_scale()); break;
                case  9: gemm< 9>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K, A.get_scale()); break;
                case  8: gemm< 8>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K, A.get_scale()); break;
                case  7: gemm< 7>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K, A.get_scale()); break;
                case  6: gemm< 6>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K, A.get_scale()); break;
                case  5: gemm< 5>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K, A.get_scale()); break;
                case  4: gemm< 4>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K, A.get_scale()); break;
                case  3: gemm< 3>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K, A.get_scale()); break;
                case  2: gemm< 2>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K, A.get_scale()); break;
                case  1: gemm< 1>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K, A.get_scale()); break;
                default: break;
            }
            // std::cout << "    N="<<N<<"/"<<N0<< std::endl;
            /*
            if (N==N0) {
                // calcule
                gemm<N0>(A.addr(0,i),B.addr(0,j),C.addr(i,j),A.LD(),B.LD(),C.LD(), K);
                return;
            }
            if constexpr (N0>1) { // arret de la recursion
                sgemm_bloc<N0-1,M0,K0>(A, B, C, i, j, N, K);
            }
            */
        }

        template<size_t K0, size_t M0>
        void mul_mat(const Matrice<TYPE_A,K0,M0,SCALE>& A, const Matrice<fp32_t>& B, Matrice<fp32_t>& C) const {
            static_assert(K0==2);
            static_assert(M0==16);
            const auto m = C.DIM1(); // == A.DIM2()
            const auto n = C.DIM2(); // == B.DIM2()
            const auto k = A.DIM1(); // == B.DIM1()
            // K0 = 2; M0 = 16; !!!
            constexpr size_t N0 = 16;
            constexpr size_t M1 =  2;

            //#pragma omp parallel for private(B_cache) schedule(guided)
            // bool premier = true;
            //#pragma omp parallel for private(premier) private(B_cache) schedule(guided)
            // TODO: faire qq teste avec OpenMP
#pragma omp parallel for schedule(guided)
            for (size_t i=0; i<m; i+=M1*M0) {
                for (size_t j=0; j<n; j+=N0) {
                    const auto N = std::min(n-j,N0);
                    // TODO: premier => mettre B<bf16> en cache
#pragma GCC unroll 8
                    for (size_t i2=0; i2<M1*M0; i2+=M0) {
                        sgemm_bloc<N0,M0,K0>(A, B, C, i+i2, j, N, k);
                    }
                }
            }
        }

    };

}

//////////////////////////////////////////////////////////////////////////////////
// l'init du backend:
//------------------------------------------------------------------------------------
// le context definissant le backend
// - les tenseur??? comment on gere les tenseurs du backend?
//namespace ggml::backend::bf16::tensor {
//}
// - le buffer
namespace ggml::backend::bf16::buffer {
    //  => comme pour les autre C => C++: class context;
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
        // @ voir si on les met dans le JSON de config...
        static constexpr std::list<std::string> LIST_WEIGHT() { return {
                "fn_down.weight",
                "fn_gate.weight",
                "fn_up.weight",
                "ttn_k.weight",
                "ttn_q.weight",
                "ttn_v.weight",
                "ttn_output.weight",
        };}
        /*
//  pour les couches:
blk.0.ffn_down.weight    [14 336, 5 120] +Q8_0
blk.0.ffn_gate.weight    [5 120, 14 336] +Q8_0
blk.0.ffn_up.weight      [5 120, 14 336] +Q8_0
blk.0.attn_k.weight      [5 120, 1 024]  +Q8_0
blk.0.attn_q.weight      [5 120, 4 096]  +Q8_0
blk.0.attn_v.weight      [5 120, 1 024]  +Q8_0
blk.0.attn_output.weight [4 096, 5 120]  +Q8_0
blk.0.ffn_norm.weight    [5 120]         -F32
blk.0.attn_norm.weight   [5 120]         -F32

//  quoi d'autre?
//  ???
         */
        inline static ggml::backend::bf16::Tensor_t* get_tensor_type(const ggml_tensor * tensor) {
            for (auto t : ggml::backend::bf16::tensors ) {
                //std::cout << " ? " << t->NAME << ":" << (int)t->is_allowed(tensor) << std::endl;
                switch (t->is_allowed(tensor)) {
                case ggml::backend::bf16::Tensor_t::COMPAT::CONVERT:
                {
                    // un peu long de le laisser ici mais que les poids => fait 1 seule fois a l'init
                    const std::string name = ggml_get_name(tensor);
                    for (auto patern: LIST_WEIGHT()) {
                        if (name.find(patern) != std::string::npos) {
                            // std::cout << "transforme: " <<t->NAME<<":"<<tensor->name<< tensor << std::endl;
                            return t;
                        }
                    }
                    //std::cout << "transforme pas possible: " <<t->NAME<<"?"<<tensor->name<< tensor << std::endl;
                }
                    break;
                case ggml::backend::bf16::Tensor_t::COMPAT::NATIVE:
                    //std::cout << "transforme: " <<t->NAME<<"+"<<tensor->name<< tensor << std::endl;
                    return t;
                    break;
                }
            }
            return &ggml::backend::bf16::tensor_non_supporte_t;
        }
        // l'interface: ggml_backend_buffer_i
        inline const char * get_name(void){ return "BF16"; }
        inline void *       get_base(void) { return m_data; }
        inline void         init_tensor(ggml_tensor * tensor) {
            // on configurer tous les tenseur tel que l'on veux qu'il soit:
            auto t = ggml::backend::bf16::Tensor_t::GET(tensor);
            if (t == nullptr) {
                // pas encore vu => on va le configurer:
                ggml::backend::bf16::Tensor_t::SET(tensor, get_tensor_type(tensor));
            } else {
                std::cerr << "re-init_tensor:" <<tensor->op<< tensor << std::endl;
            }
        }
        // inline void load_tensor(struct ggml_tensor * tensor, File, offset);
        inline void set_tensor(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
            auto t = ggml::backend::bf16::Tensor_t::GET(tensor);
            if (t) {
                t->set_tensor(tensor, data, offset, size);
            } else {
                // ne devrai pas arriver!!!
                std::cerr << "set_tensor not configured:" <<tensor->op<<"::"<<tensor->name<< tensor << std::endl;
                memcpy((char *)tensor->data + offset, data, size);
            }
            return;
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
            //return ggml::backend::bf16::buffer::TENSOR_ALIGNMENT; // @ voir quoi mettre: 512 bits?
            return TENSOR_ALIGNMENT; // @ voir quoi mettre: 512 bits?
        }
        inline size_t get_max_size(){
            return (size_t)64*1024*1024*1024; // la taille de la RAM/GGT/VRAM ???
        }
        inline size_t get_alloc_size(const struct ggml_tensor * tensor) {
            // Voir si on peu utiliser le type... est-t'on sur de ce qu'il va etre au final?
            //ggml::backend::bf16::buffer::ctx(tensor->buffer);
            auto type = ggml::backend::bf16::buffer::context::get_tensor_type(tensor);
            //std::cout << "get_alloc_size("<<tensor->name<<"):" << ((type==nullptr)?0:type->get_alloc_size(tensor)) << "/" << ggml_nbytes(tensor) << std::endl;
            if (type) {
                return type->get_alloc_size(tensor);
            }
            std::cout << "??? get_alloc_size("<<tensor->name<<"):" << "/" << ggml_nbytes(tensor) << std::endl;
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
                    ggml::backend::bf16::op::GET(node)->exec(node);
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
            //auto op_base = ggml_bf16_op::GET(op); // tjs null, le graph est tjs re-caculé
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
                        ggml::backend::bf16::op::SET(op, imp);
                        return true;
                        //auto bf16_op = imp->inst(op);
                        //if (bf16_op) {
                        //    ggml::backend::bf16::op::SET(op, bf16_op);
                        //    return true;
                        //}
                    }
                }
                return false;
            } else if (op->op == GGML_OP_NONE) {
                // OK deplacé dans l'init, ca devrait etre bon
                if (! ggml::backend::bf16::Tensor_t::GET(op)) {
                    // y a t'il des cas ou c'est encore possible???
                    std::cerr << "ERREUR: tenseur non configuré: " <<op->op<<op<< std::endl;
                }
                // TODO: ca ne sert a rien mais on peu retourner true si le poid est pour nous
                return false;
            }
            // else if (op->op == GGML_OP_OUT_PROD) {
            //    std::cout << " > " <<op->op<<"("<<log_srcs(op)<<" => "<<op<<"): "<< op->name<<std::endl;
            //}
            //if (op_base==nullptr) {
            //std::cout << "TODO"<<(void*)op<<"/"<<op->bf16_op;
            //ggml_bf16_op::SET(op, new ggml_bf16_op_notsupported());
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

    // + voir si on ne cre pas une map?vector "correcte" avec!
    // et gerer ca avec des regex: https://en.cppreference.com/w/cpp/regex/regex_match
    // "prefix\\.[:digit:]+\\.suffix\\.weight"
    // la liste des "backend" supporté (enfin de format des tenseurs de poids)
    //  https://json.nlohmann.me/features/enum_conversion/
    enum class BACKEND_TYPE {
        // [jart] kernel
        BF16_32x1_5x5,
        BF16_32x1_4x6,
        E4M3_32x1_G_5x5,
        E4M3_32x1_G_4x6,
        E4M3_32x1_C_5x5,
        E4M3_32x1_C_4x6,
        // mon kernel
        BF16_2x16,
        E4M3_2x16_G,
        E4M3_2x16_C,
        INVALID
    };
    NLOHMANN_JSON_SERIALIZE_ENUM( BACKEND_TYPE, {
            {BACKEND_TYPE::INVALID, nullptr},
            {BACKEND_TYPE::BF16_32x1_5x5, "BF16_32x1_5x5"},
            {BACKEND_TYPE::BF16_32x1_4x6, "BF16_32x1_4x6"},
            {BACKEND_TYPE::E4M3_32x1_G_5x5, "FP8_E4M3G_32x1_5x5"},
            {BACKEND_TYPE::E4M3_32x1_G_4x6, "FP8_E4M3G_32x1_4x6"},
            {BACKEND_TYPE::E4M3_32x1_C_5x5, "FP8_E4M3C_32x1_5x5"},
            {BACKEND_TYPE::E4M3_32x1_C_4x6, "FP8_E4M3C_32x1_4x6"},

            {BACKEND_TYPE::BF16_2x16, "BF16_2x16"},
            {BACKEND_TYPE::E4M3_2x16_G, "FP8_E4M3G_2x16"},
            {BACKEND_TYPE::E4M3_2x16_C, "FP8_E4M3C_2x16"},
    })
#define DECODE_TYPE(val) val.template get<BACKEND_TYPE>()


    static void init() {
        static bool first = true;
        if (!first) return;
        first = false;

        //std::string backend_config{getenv("GGML_USE_BACKEND_BF16")};
        json backend_config = json::parse(getenv("GGML_USE_BACKEND_BF16"));
        //  '[ "BF16_32x1", ... ]'
        std::cout << backend_config.dump(4) << std::endl;
        ggml::backend::bf16::matmul_ops.clear();

        json j;

        std::vector<BACKEND_TYPE> backend_list = backend_config;
        for (auto b : backend_list) {
            j["backend"] = b;
            std::cout << "BF16 register:" << j << std::endl;
            switch (b) {
            case BACKEND_TYPE::BF16_32x1_5x5:
                // les "JART" en dernier de preferance.
                ggml::backend::bf16::tensors.push_back(ggml::backend::bf16::tensor_type<bf16_t,32>());
                ggml::backend::bf16::matmul_ops.push_back(new ggml_bf16_op_matmul_tg<bf16_t, Scale::NONE, 8, 3>);
                ggml::backend::bf16::matmul_ops.push_back(new ggml_bf16_op_matmul_pp<bf16_t, Scale::NONE, 5, 5>);
                break;
            case BACKEND_TYPE::BF16_32x1_4x6:
                // les "JART" en dernier de preferance.
                ggml::backend::bf16::tensors.push_back(ggml::backend::bf16::tensor_type<bf16_t,32>());
                ggml::backend::bf16::matmul_ops.push_back(new ggml_bf16_op_matmul_tg<bf16_t, Scale::NONE, 8, 3>);
                ggml::backend::bf16::matmul_ops.push_back(new ggml_bf16_op_matmul_pp<bf16_t, Scale::NONE, 4, 6>);
                break;
            case BACKEND_TYPE::E4M3_32x1_G_5x5:
                ggml::backend::bf16::tensors.push_back(ggml::backend::bf16::tensor_type<f8_E4M3_t,32,1,Scale::GLOBAL>());
                ggml::backend::bf16::matmul_ops.push_back(new ggml_bf16_op_matmul_tg<f8_E4M3_t, Scale::GLOBAL, 8, 3>);
                ggml::backend::bf16::matmul_ops.push_back(new ggml_bf16_op_matmul_pp<f8_E4M3_t, Scale::GLOBAL, 5, 5>);
                break;
            case BACKEND_TYPE::E4M3_32x1_G_4x6:
                ggml::backend::bf16::tensors.push_back(ggml::backend::bf16::tensor_type<f8_E4M3_t,32,1,Scale::GLOBAL>());
                ggml::backend::bf16::matmul_ops.push_back(new ggml_bf16_op_matmul_tg<f8_E4M3_t, Scale::GLOBAL, 8, 3>);
                ggml::backend::bf16::matmul_ops.push_back(new ggml_bf16_op_matmul_pp<f8_E4M3_t, Scale::GLOBAL, 4, 6>);
                break;
            case BACKEND_TYPE::E4M3_32x1_C_5x5:
                ggml::backend::bf16::tensors.push_back(ggml::backend::bf16::tensor_type<f8_E4M3_t,32,1,Scale::PER_COL>());
                ggml::backend::bf16::matmul_ops.push_back(new ggml_bf16_op_matmul_tg<f8_E4M3_t, Scale::PER_COL, 8, 3>);
                ggml::backend::bf16::matmul_ops.push_back(new ggml_bf16_op_matmul_pp<f8_E4M3_t, Scale::PER_COL, 5, 5>);
                break;
            case BACKEND_TYPE::E4M3_32x1_C_4x6:
                ggml::backend::bf16::tensors.push_back(ggml::backend::bf16::tensor_type<f8_E4M3_t,32,1,Scale::PER_COL>());
                ggml::backend::bf16::matmul_ops.push_back(new ggml_bf16_op_matmul_tg<f8_E4M3_t, Scale::PER_COL, 8, 3>);
                ggml::backend::bf16::matmul_ops.push_back(new ggml_bf16_op_matmul_pp<f8_E4M3_t, Scale::PER_COL, 4, 6>);
                break;

            case BACKEND_TYPE::BF16_2x16:
                ggml::backend::bf16::tensors.push_back(ggml::backend::bf16::tensor_type<bf16_t,2,16>());
                ggml::backend::bf16::matmul_ops.push_back(new ggml::bf16::op_matmul::bf16_2x16<bf16_t, Scale::NONE>);
                break;
            case BACKEND_TYPE::E4M3_2x16_G:
                ggml::backend::bf16::tensors.push_back(ggml::backend::bf16::tensor_type<f8_E4M3_t,2,16,Scale::GLOBAL>());
                ggml::backend::bf16::matmul_ops.push_back(new ggml::bf16::op_matmul::bf16_2x16<f8_E4M3_t, Scale::GLOBAL>);
                break;
                // ...
            case BACKEND_TYPE::INVALID:
                std::cout << " > ERREUR: BF16_Backend non connu"<< std::endl;
                break;
            default:
                std::cout << " > ERREUR: Backend non implementé"<< std::endl;
            }
        }
        // ceux tjs "possible"...
        ggml::backend::bf16::tensors.push_back(ggml::backend::bf16::tensor_type<fp32_t,32>());
        ggml::backend::bf16::tensors.push_back(&ggml::backend::bf16::tensor_non_supporte_t);

        for (auto t : ggml::backend::bf16::tensors) {
            std::cout << "BF16 register <tensor_type>: " << t->NAME << std::endl;
        }
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
    // OK il faut le creer...
    ggml::backend::bf16::init();
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
