// new kernel for test.
//  - use a base bloc on: A[M/16][K/2][16][2]
//  - C is compute with bloc: C[N0][16]

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
