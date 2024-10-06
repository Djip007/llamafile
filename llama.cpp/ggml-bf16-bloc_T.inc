namespace ggml::bf16::op_matmul {

    template<typename TYPE_A,
    std::size_t M0=16, std::size_t N0=16, std::size_t K0=2, std::size_t K1=1024
    >
    class bf16_2x16_T: public ggml::backend::bf16::op {
    public:
        using type_A = TYPE_A;
        using type_B = fp32_t;
        using type_C = fp32_t;
        using tensorA_t = ggml::backend::bf16::tensor<type_A,K0,M0,K1>;
        using tensorB_t = ggml::backend::bf16::tensor<type_B,K1>;
        using tensorC_t = ggml::backend::bf16::tensor<type_C,M0>;

        using matA_t = tensorA_t::matrice_t;
        using matB_t = tensorB_t::matrice_t;
        using matC_t = tensorC_t::matrice_t;

        using type_AB = bf16_t;
        static constexpr int VS=32;

        bf16_2x16_T() {
            static_assert(K1%K0==0);
            static_assert(K0==2);  // pour l'instant pas d'autre cas pour calcul en bf16 / 1 si calcul en fp32 ...
            static_assert(M0==16); // 16xFP32 => 1 AVX512 / 8xFP32 => 1 AVX256 ...
        }
        void exec(struct ggml_tensor *op) const override {
            // normalement ici les type sont deja controlé
            const matA_t A(op->src[0]);
            const matB_t B(op->src[1]);
            matC_t C(op);
            mul_mat(A, B, C);
        }

        bool B_is_allowed(const struct ggml_tensor *B) override {
            return tensorB_t::type()->is_allowed(B) == ggml::backend::bf16::Tensor_t::COMPAT::NATIVE;
        }
        bool C_is_allowed(const struct ggml_tensor *C) override {
            return tensorC_t::type()->is_allowed(C) == ggml::backend::bf16::Tensor_t::COMPAT::NATIVE;
        }
        bool A_is_allowed(const struct ggml_tensor *A) override {
            auto a = ggml::backend::bf16::Tensor_t::GET(A);
            if(!a && A->view_src) { return false; }
            GGML_ASSERT(a!=nullptr);
            if (a == tensorA_t::type()) return true;
            return false;
        }

    private:

        static inline auto load_B(const type_AB* pB)          { return (__m512bh)_mm512_set1_ps(*(float*)(pB)); }
        static inline auto load_A(const type_A* pA)           { return load<typename type_t<type_AB,M0*K0>::t>(pA); }
        static inline void init_C(type_t<type_C,M0>::t& C)    { C = _mm512_setzero_ps(); }
        static inline type_t<type_C,M0>::t load_C(type_C* pC) { return load<typename type_t<type_C,M0>::t>(pC); }

        template<size_t N>
        inline void gemm(type_C pC[N0][M0], const type_A *pA, const type_AB pB[N0][K1]) const {
            static_assert(N>0);
            typename type_t<type_C,M0>::t C[N0];

            for(size_t j=0; j<N; j++) {
                init_C(C[j]);
            }
            for (size_t k0=0; k0<K1; k0+=K0) {
                auto A = load_A(pA + (k0)*M0); // == ~TA[M0][K0]
                for (size_t j=0; j<N; ++j) {
                    auto B = load_B(&pB[j][k0]);
                    C[j] = madd(C[j], A, B);
                }
            }
            for(size_t j=0; j<N; j++) {
                auto c = load_C(pC[j]);
                c += C[j];
                store(pC[j], c);
            }
        }

        inline void bloc_B(type_AB B_bloc[K1], const type_B* B) const {
            for (size_t k=0; k<K1; k+=VS) {
                auto b = load<typename type_t<type_AB,VS>::t>(B+k);
                store(&B_bloc[k],b);
            }
        }

        inline void sgemm_bloc(type_C C[N0][M0], const type_A* A, const type_AB B[N0][K1], size_t N) const {
            static_assert(N0<=16);
            switch (N) {
            case 16: gemm<16>(C, A, B); break;
            case 15: gemm<15>(C, A, B); break;
            case 14: gemm<14>(C, A, B); break;
            case 13: gemm<13>(C, A, B); break;
            case 12: gemm<12>(C, A, B); break;
            case 11: gemm<11>(C, A, B); break;
            case 10: gemm<10>(C, A, B); break;
            case  9: gemm< 9>(C, A, B); break;
            case  8: gemm< 8>(C, A, B); break;
            case  7: gemm< 7>(C, A, B); break;
            case  6: gemm< 6>(C, A, B); break;
            case  5: gemm< 5>(C, A, B); break;
            case  4: gemm< 4>(C, A, B); break;
            case  3: gemm< 3>(C, A, B); break;
            case  2: gemm< 2>(C, A, B); break;
            case  1: gemm< 1>(C, A, B); break;
            default: break;
            }
        }

        void mul_mat(const matA_t& A, const matB_t& B, matC_t& C) const {
            constexpr size_t N1 = 8; // 1; //4; //8;
            constexpr size_t M1 = 1; // 1; //4; //8;

            const auto m = C.DIM1(); // == A.DIM2()
            const auto n = C.DIM2(); // == B.DIM2()
            const auto k = A.DIM1(); // == B.DIM1()
            GGML_ASSERT(m == A.DIM2());
            GGML_ASSERT(n == B.DIM2());
            GGML_ASSERT(k == B.DIM1());
            GGML_ASSERT(k%(K0) == 0);
            GGML_ASSERT(K1==0 || k%(K1) == 0);
            GGML_ASSERT(m%(M0*M1) == 0);

            // cas pour N=1 ... utilisé pour le "tg"
            // + cas pour N<16 ???
            if (n==0) {
                type_C C_bloc[N0][M0];
                type_AB B_cache[N0][K1];

                for (size_t k2=0; k2<k; k2+=K1) {
                    bloc_B(B_cache[0], B.addr(k2,0));

                    for (size_t i1=0; i1<m; i1+=M0) {
                        if (k2==0) {
                            for (int i0=0; i0<N0; i0++) C_bloc[0][i0]=0;
                        } else {
                            for (int i0=0; i0<N0; i0++) C_bloc[0][i0]=C.addr(i1,0)[i0];
                        }
                        // calcul du kernel.
                        gemm<1>(C_bloc, A.addr(k2,i1), B_cache);
                        // stokage de C
                        for (int i0=0; i0<M0; i0++) C.addr(i1,0)[i0]=C_bloc[0][i0];
                    }
                }
            } else {
                type_C C_bloc[N0][M0];
                type_AB B_cache[N1][N0][K1];

                for (size_t k2=0; k2<k; k2+=K1) {
                    for (size_t j2=0; j2<n; j2+=N1*N0) {
                        const auto N1_MAX=std::min(N1*N0,(n-j2));
                        for (size_t j=0; j<N1_MAX; j++) {
                            auto j1 = j/N0;
                            auto j0 = j%N0;
                            bloc_B(B_cache[j1][j0], B.addr(k2,j2+j));
                        }
                        for (size_t i2=0; i2<m; i2+=M1*M0) { // parcours de bloc A[M0][K1]
                            for (size_t j1=0; j1<N1_MAX; j1+=N0) { // parcours de bloc B[N0][K1]
                                auto N = std::min(N0,N1_MAX-j1);
                                for (size_t i1=0; i1<M1*M0; i1+=M0) {
                                    // recuperation de C
                                    if (k2==0) {
                                        for (int j0=0; j0<N; j0++)
                                            for (int i0=0; i0<M0; i0++)
                                                C_bloc[j0][i0]=0;
                                    } else {
                                        for (int j0=0; j0<N; j0++) {
                                            auto pC = C.addr(i2+i1,j2+j1+j0);
                                            for (int i0=0; i0<M0; i0++)
                                                C_bloc[j0][i0]=pC[i0];
                                        }
                                    }
                                    // calcul du kernel.
                                    sgemm_bloc(C_bloc, A.addr(k2,i2+i1), B_cache[j1/N0], N);
                                    // stokage de C
                                    for (int j0=0; j0<N; j0++) {
                                        auto pC = C.addr(i2+i1,j2+j1+j0);
                                        for (int i0=0; i0<M0; i0++)
                                            pC[i0]=C_bloc[j0][i0];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };

}
