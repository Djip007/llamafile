// new kernel for test.
//  - use a base bloc on: A[M/16][K/2][16][2]
//  - C is compute with bloc: C[N0][16]

namespace ggml::bf16::op_matmul {

    //CORRECTION
    template<typename T1, typename T2> T2 correct(T2 val);

    // __m512
    template<> __m512 correct<fp32_t,   __m512>(__m512 val) { return val; }
    template<> __m512 correct<bf16_t,   __m512>(__m512 val) { return val; }
    template<> __m512 correct<fp16_t,   __m512>(__m512 val) { return val; }
    template<> __m512 correct<f8_E4M3_t,__m512>(__m512 val) { return val*CORRECTION<f8_E4M3_t>(); }

    template<typename TYPE_A>
    class bf16_2x16_NONE: public ggml::backend::bf16::op {
    public:
        static constexpr size_t K0 =  2;
        static constexpr size_t M0 = 16;
        static constexpr size_t K1 =  8; // des blocks de 4 (8/2) pour ameloré les lecture de B!
        using tensorA_t = ggml::backend::bf16::tensor<TYPE_A,K0,M0>;
        using tensorB_t = ggml::backend::bf16::tensor<fp32_t,8>;
        using tensorC_t = ggml::backend::bf16::tensor<fp32_t,16>;
        using matA_t = tensorA_t::matrice_t;
        using matB_t = tensorB_t::matrice_t;
        using matC_t = tensorC_t::matrice_t;

        void exec(struct ggml_tensor *op) const override {
#ifdef DO_TIMING
            mesure time; time.start();
#endif
            // normalement ici les type sont deja controlé
            const matA_t A(op->src[0]);
            const matB_t B(op->src[1]);
            matC_t       C(op);
            mul_mat(A, B, C);
#ifdef DO_TIMING
            auto dt = time.end();
            std::cout << " bf16_2x16_G> " <<op->op<<"("<<log_srcs(op)<<" => "<<op<<"): "<< op->name << " => "
                    << dt*1000000 << " us / "
                    << (1e-9*2*op->ne[0]*op->ne[1]*op->src[1]->ne[0])/dt << " GFlops/s"
                    << std::endl;
#endif
        }

        // pour l'instant tjs des type FP32 natif ... pas le choix et y en a pas d'autre!
        bool C_is_allowed(const struct ggml_tensor *C) override {
            return tensorC_t::type()->is_allowed(C) == ggml::backend::bf16::Tensor_t::COMPAT::NATIVE;
        }
        bool B_is_allowed(const struct ggml_tensor *B) override {
            return tensorB_t::type()->is_allowed(B) == ggml::backend::bf16::Tensor_t::COMPAT::NATIVE;
        }
        bool A_is_allowed(const struct ggml_tensor *A) override {
            auto a = ggml::backend::bf16::Tensor_t::GET(A);
            if(!a && A->view_src) {
                // une vue => pas "encore" supporté!
                return false;
            }
            GGML_ASSERT(a!=nullptr);
            // on sais deja:
            if (a == tensorA_t::type()) return true;
            return false;
        }

    private:
        // le bloc de bas niveau
        template<size_t N0, typename TA, typename TB, typename TC>
        static void gemm(const TA *pA, const TB *pB, TC *pC, std::size_t ldb, std::size_t ldc, std::size_t K2) { //, float scale) {
            // lda: comment passer de A[k,i] => A[k,i+M1]
            // ldb: comment passer de B[k,j] => B[k,j+1]
            // ldc: comment passer de C[i,j] => C[i,j+1]
            //constexpr int K0 =  2; //
            //constexpr int M0 = 16; // 16 FP32 => 1 AVX512!
            //constexpr int K1 =  8; // des blocks de 4 (8/2) pour ameloré les lecture de B!
            static_assert(N0>0);
            GGML_ASSERT(K2%K1 == 0);

            __m512   C[N0];    // m512   == fp32[M0]
            __m512bh A[K1/K0]; // m512bh == bf16[M0][K0]

#pragma GCC unroll N0
            for(size_t j=0; j<N0; j++) {
                C[j] = _mm512_setzero_ps();
            }

            for (size_t k2=0; k2<K2; k2+=K1) { // de 8 en 8 ...
                // chargement de A
#pragma GCC unroll K1
                for (size_t k1=0; k1<K1/K0; ++k1) {  // [0..3]
                    A[k1] = load<__m512bh>(pA + k2*M0 + k1*M0*K0); // lda == K2*M0 ...
                }
#pragma GCC unroll N0
                for (size_t j=0; j<N0; ++j) {  // [0..~16]
                    // on charge K1 valeur de B
                    __m128bh B = load<__m128bh>(pB+j*ldb+k2);
#pragma GCC unroll K1
                    for (size_t k1=0; k1<K1/K0; ++k1) {  // [0..4]
                        auto _B = _mm512_broadcastd_2pbh(B);
                        B = _mm_shiftl_2pbh(B);
                        C[j] = _mm512_dpbf16_ps(C[j], A[k1], _B);
                    }
                }
            }

            // ecriture de C...
#pragma GCC unroll N0
            for(size_t j=0; j<N0; j++) {
                store(pC+j*ldc, correct<TA>(C[j]));
                //store(pC+j*ldc, C[j]);
            }
        }

        template<size_t N0, size_t M0, size_t K0>
        inline void sgemm_bloc(const matA_t& A, const matB_t& B, matC_t& C, size_t i, size_t j, size_t N, size_t K) const {
            static_assert(N0<=16);
            switch (N) {
                case 16: gemm<16>(A.addr(0,i),B.addr(0,j),C.addr(i,j),B.LD(),C.LD(), K); break;
                case 15: gemm<15>(A.addr(0,i),B.addr(0,j),C.addr(i,j),B.LD(),C.LD(), K); break;
                case 14: gemm<14>(A.addr(0,i),B.addr(0,j),C.addr(i,j),B.LD(),C.LD(), K); break;
                case 13: gemm<13>(A.addr(0,i),B.addr(0,j),C.addr(i,j),B.LD(),C.LD(), K); break;
                case 12: gemm<12>(A.addr(0,i),B.addr(0,j),C.addr(i,j),B.LD(),C.LD(), K); break;
                case 11: gemm<11>(A.addr(0,i),B.addr(0,j),C.addr(i,j),B.LD(),C.LD(), K); break;
                case 10: gemm<10>(A.addr(0,i),B.addr(0,j),C.addr(i,j),B.LD(),C.LD(), K); break;
                case  9: gemm< 9>(A.addr(0,i),B.addr(0,j),C.addr(i,j),B.LD(),C.LD(), K); break;
                case  8: gemm< 8>(A.addr(0,i),B.addr(0,j),C.addr(i,j),B.LD(),C.LD(), K); break;
                case  7: gemm< 7>(A.addr(0,i),B.addr(0,j),C.addr(i,j),B.LD(),C.LD(), K); break;
                case  6: gemm< 6>(A.addr(0,i),B.addr(0,j),C.addr(i,j),B.LD(),C.LD(), K); break;
                case  5: gemm< 5>(A.addr(0,i),B.addr(0,j),C.addr(i,j),B.LD(),C.LD(), K); break;
                case  4: gemm< 4>(A.addr(0,i),B.addr(0,j),C.addr(i,j),B.LD(),C.LD(), K); break;
                case  3: gemm< 3>(A.addr(0,i),B.addr(0,j),C.addr(i,j),B.LD(),C.LD(), K); break;
                case  2: gemm< 2>(A.addr(0,i),B.addr(0,j),C.addr(i,j),B.LD(),C.LD(), K); break;
                case  1: gemm< 1>(A.addr(0,i),B.addr(0,j),C.addr(i,j),B.LD(),C.LD(), K); break;
                default: break;
            }
        }

        void mul_mat(const matA_t& A, const matB_t& B, matC_t& C) const {
            const auto m = C.DIM1(); // == A.DIM2()
            const auto n = C.DIM2(); // == B.DIM2()
            const auto k = A.DIM1(); // == B.DIM1()
            // K0 = 2; M0 = 16; !!!
            constexpr size_t N0 = 16;
            constexpr size_t M1 =  4;

#pragma omp parallel for schedule(guided)
            for (size_t i=0; i<m; i+=M1*M0) {
                for (size_t j=0; j<n; j+=N0) {
                    const auto N = std::min(n-j,N0);
                    for (size_t i2=0; i2<M1*M0; i2+=M0) {
                        sgemm_bloc<N0,M0,K0>(A, B, C, i+i2, j, N, k);
                    }
                }
            }
        }

    };

}
