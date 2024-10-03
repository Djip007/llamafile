// new kernel for test.
//  - use a base bloc on: A[M/16][K/2][16][2]
//  - C is compute with bloc: C[N0][16]
// 1024 2048 4096 5120  2560  K1=4
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
//   - K1 : A en L1  => 32k: /16 = 2048 max
//   - M  : tous les M  => dispache par coeurs
//   - N1 : B_cache en L2 (fp32->bf16) => 1M  /(16*2*2048) = 16 max
//   - N  : tous les N
//   - K  : tous les K

namespace ggml::bf16::op_matmul {
    // TODO gerer les autres types
    //  - COMPUTE_TYPE ACC_TYPE ? FP32 += FP16*FP16 ...
    //  - A_TYPE
    //  - C_TYPE / B_TYPE ... pour l'instant tjs fp32 ?
    template<typename TYPE_A, Scale SCALE, size_t N0=16, size_t K1=1024, size_t M0=16, size_t K0=2>
    class bf16_2x16: public ggml::backend::bf16::op {
    public:
        bf16_2x16() {
            static_assert(K1%K0==0);
            static_assert(K0==2);  // pour l'instant pas d'autre cas pour calcul en bf16 / 1 si calcul en fp32 ...
            static_assert(M0==16); // 16xFP32 => 1 AVX512 / 8xFP32 => 1 AVX256 ...
        }
        void exec(struct ggml_tensor *op) const override {
#ifdef DO_TIMING
            mesure time; time.start();
#endif
            // normalement ici les type sont deja controlé
            const Matrice<TYPE_A,K0,M0,K1,SCALE> A(op->src[0]);
            const Matrice<fp32_t,K1>             B(op->src[1]);
            Matrice<fp32_t,M0>                   C(op);
            mul_mat(A, B, C);
#ifdef DO_TIMING
            auto dt = time.end();
            std::cout << " bf16_2x16> " <<op->op<<"("<<log_srcs(op)<<" => "<<op<<"): "<< op->name << " => "
                    << dt*1000000 << " us / "
                    << (1e-9*2*op->ne[0]*op->ne[1]*op->src[1]->ne[0])/dt << " GFlops/s"
                    << std::endl;
#endif
        }

        bool B_is_allowed(const struct ggml_tensor *B) override {
            return ggml::backend::bf16::tensor_type<fp32_t,K1>()->is_allowed(B) == ggml::backend::bf16::Tensor_t::COMPAT::NATIVE;
        }
        bool C_is_allowed(const struct ggml_tensor *C) override {
            return ggml::backend::bf16::tensor_type<fp32_t,M0>()->is_allowed(C) == ggml::backend::bf16::Tensor_t::COMPAT::NATIVE;
        }
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
            if (a == ggml::backend::bf16::tensor_type<TYPE_A,K0,M0,K1,SCALE>()) return true;
            return false;
        }

    private:
        // le bloc de bas niveau
        // TODO ajouter in gemv (mat/vect): N=N0=1 !!
        // TA pour l'instant bf16 / fp8_...
        // gerer le scale comme un variatic template?
        template<size_t N, typename TB=bf16_t, typename TC=fp32_t>
        inline void gemm(type_t<TC,M0>::t C[N0], const TYPE_A *pA, const TB pB[N0][K1], const type_t<TC,M0>::t& scale) const {
            // pA[:][M0][K0]
            using TAC = typename type_t<bf16_t,M0*K0>::t;
            //using TBC = typename type_t<bf16_t,K1*K0>::t;
            //using TBC = typename type_t<bf16_t,M0*K0>::t; // broadcast de bf16_t[K0]
            static_assert(N>0);

#pragma GCC unroll N
            for(size_t j=0; j<N; j++) {
                // TODO: mise a 0 generique
                C[j] = _mm512_setzero_ps();
            }

#pragma GCC unroll 8
            for (size_t k2=0; k2<K1; k2+=K0) {
                // chargement de A
                TAC A = load<TAC>(pA + k2*M0); // == ~TA[M0][K0]
#pragma GCC unroll N
                for (size_t j=0; j<N; ++j) {
                    // on charge K1 valeur de B
                    //auto B = load<TBC>(&pB[j][k2]);
                    //auto B = broadcast_2x(pB[j], );
                    //auto B = broadcast<bf16_t,8,K0>(&pB[j][k2]);
                    auto B = (__m512bh)_mm512_set1_ps(*(float*)(&pB[j][k2]));
                    C[j] = madd(C[j], A, B);
                }
            }

            // ecriture de C...
#pragma GCC unroll N
            for(size_t j=0; j<N; j++) {
                //if (SCALE!=Scale::NONE)    C[j] *= scale;
                //if (SCALE==Scale::PER_COL) C[j] *= scale;
                static_assert(SCALE!=Scale::NONE);
                C[j] *= scale;
            }
        }

        template<typename TB, int KB=32>
        inline void bloc_B(bf16_t B_bloc[K1], const TB* B) const {
            // KB la taille du vecteur de calcul : celuis de l'AVX utilisé
            for (size_t k=0; k<K1; k+=KB) {
                auto b = load<typename type_t<bf16_t,KB>::t>(B+k);
                store(&B_bloc[k],b);
            }
        }

        // TODO: gerer tous les types...
        inline void sgemm_bloc(type_t<fp32_t,M0>::t C[N0], const TYPE_A* A, const bf16_t B[N0][K1], size_t N, const type_t<fp32_t,M0>::t& scale) const {
            static_assert(N0<=16);
            switch (N) {
            case 16: gemm<16>(C, A, B, scale); break;
            case 15: gemm<15>(C, A, B, scale); break;
            case 14: gemm<14>(C, A, B, scale); break;
            case 13: gemm<13>(C, A, B, scale); break;
            case 12: gemm<12>(C, A, B, scale); break;
            case 11: gemm<11>(C, A, B, scale); break;
            case 10: gemm<10>(C, A, B, scale); break;
            case  9: gemm< 9>(C, A, B, scale); break;
            case  8: gemm< 8>(C, A, B, scale); break;
            case  7: gemm< 7>(C, A, B, scale); break;
            case  6: gemm< 6>(C, A, B, scale); break;
            case  5: gemm< 5>(C, A, B, scale); break;
            case  4: gemm< 4>(C, A, B, scale); break;
            case  3: gemm< 3>(C, A, B, scale); break;
            case  2: gemm< 2>(C, A, B, scale); break;
            case  1: gemm< 1>(C, A, B, scale); break;
            default: break;
            }
        }

        void mul_mat(const Matrice<TYPE_A,K0,M0,K1,SCALE>& A, const Matrice<fp32_t,K1>& B, Matrice<fp32_t,M0>& C) const {
            constexpr size_t N1 = 4; // 1; //4; //8;
            const auto m = C.DIM1(); // == A.DIM2()
            const auto n = C.DIM2(); // == B.DIM2()
            const auto k = A.DIM1(); // == B.DIM1()
            GGML_ASSERT(m == A.DIM2());
            GGML_ASSERT(n == B.DIM2());
            GGML_ASSERT(k == B.DIM1());

            // cas pour N=1 ... utilisé pour le "tg"
            if (n==1) {
                bf16_t B_cache[1][K1];
                typename type_t<fp32_t,M0>::t C_bloc[N0];

                // parallele region
#pragma omp parallel private(C_bloc) num_threads(4)
                for (size_t k2=0; k2<k; k2+=K1) {
#pragma omp master
//#pragma omp single
                    {
                        bloc_B(B_cache[0], B.addr(k2,0));
                    }
#pragma omp for schedule(dynamic, 16)
                    for (size_t i1=0; i1<m; i1+=M0) {
                        // calcul du kernel.
                        gemm<1>(C_bloc, A.addr(k2,i1), B_cache, A.template get_scale_v<M0>(i1));
                        // stokage de C
                        if (k2==0) {
                            store(C.addr(i1,0),C_bloc[0]);
                        } else {
                            auto c = load<typename type_t<fp32_t,M0>::t>(C.addr(i1,0));
                            c += C_bloc[0];
                            store(C.addr(i1,0),c);
                        }
                    }
                }
            } else {
                typename type_t<fp32_t,M0>::t C_bloc[N0];
                bf16_t B_cache[N1][N0][K1];

                // parallele region
#pragma omp parallel private(C_bloc)
                for (size_t k2=0; k2<k; k2+=K1) {
                    for (size_t j2=0; j2<n; j2+=N1*N0) {
                        const auto N1_MAX=std::min(N1*N0,(n-j2));
//#pragma omp for schedule(guided)
#pragma omp for schedule(dynamic, 2)
                        for (size_t j=0; j<N1_MAX; j++) {
                            auto j1 = j/N0;
                            auto j0 = j%N0;
                            //std::cout << "bloc_B" << std::endl;
                            bloc_B(B_cache[j1][j0], B.addr(k2,j2+j));
                        }
                        //#pragma omp barrier
#pragma omp for schedule(dynamic, 8)
//#pragma omp for schedule(static, 8)
//#pragma omp for schedule(guided)
                        //#pragma omp parallel for private(C_bloc) shared(B_cache) schedule(guided)
                        // TODO: bloc sur M ...
                        for (size_t i1=0; i1<m; i1+=M0) {  // si on veux // sur plus de CPU il faut decouper M... et // M2/N2
                            for (size_t j1=0; j1<N1_MAX; j1+=N0) { // parcours de bloc B[N0][K1]
                                auto N = std::min(N0,N1_MAX-j1);
                                // calcul du kernel.
                                sgemm_bloc(C_bloc, A.addr(k2,i1), B_cache[j1/N0], N, A.template get_scale_v<M0>(i1));
                                // stokage de C
                                for (size_t j0=0; j0<N; j0++) {
                                    if (k2==0) {
                                        store(C.addr(i1,j2+j1+j0),C_bloc[j0]);
                                    } else {
                                        auto c = load<typename type_t<fp32_t,M0>::t>(C.addr(i1,j2+j1+j0));
                                        c += C_bloc[j0];
                                        store(C.addr(i1,j2+j1+j0),c);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        //#pragma omp parallel for private(B_cache) schedule(guided)
        //#pragma omp parallel for private(premier) private(B_cache) schedule(guided)
        //#pragma omp parallel for schedule(guided)
        //#pragma GCC unroll 8

    };

}
