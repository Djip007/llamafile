// new kernel for test.
//  - use a base bloc on: A[M/16][K/2][16][2]
//  - C is compute with bloc: C[N0][16]
// 1024 2048 4096 5120  2560  K1=4
namespace ggml::bf16::op_matmul {
    template<typename TYPE_A, Scale SCALE, size_t N0=16, size_t M0=16, size_t K2=2048, size_t K1=1, size_t K0=2>
    class bf16_2x16_2: public ggml::backend::bf16::op {
    public:
        bf16_2x16_2() {
            static_assert(K2%(K0*K1)==0);
            static_assert(K0==2);  // pour l'instant pas d'autre cas pour calcul en bf16 / 1 si calcul en fp32 ...
            static_assert(M0==16); // 16 FP32 => 1 AVX512!
            //static_assert(K1==4);  // pas d'autre cas pour l'instant...
        }
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
        void exec(struct ggml_tensor *op) const override {
#ifdef DO_TIMING
            mesure time; time.start();
#endif
            // normalement ici les type sont deja controlé
            const Matrice<TYPE_A,K0,M0,SCALE> A(op->src[0]);
            const Matrice<fp32_t,K0*K1>       B(op->src[1]);
            Matrice<fp32_t,M0>                C(op);
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
            return ggml::backend::bf16::tensor_type<fp32_t,K0*K1>()->is_allowed(B) == ggml::backend::bf16::Tensor_t::COMPAT::NATIVE;
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
            if (a == ggml::backend::bf16::tensor_type<TYPE_A,2,16,SCALE>()) return true;
            return false;
        }

    private:
        // le bloc de bas niveau
        // TA pour l'instant bf16 / fp8_...
        template<size_t N, typename TA, typename TB=bf16_t, typename TC=fp32_t>
        //static void gemm(const TA *pA, const TB pB[N0][K1], TC pC[N0][M0], std::size_t K, float scale) {
        //static void gemm(const TA *pA, const TB pB[N0][K1], type_t<TC,M0>::t pC[N0], std::size_t K, const TS scale) {
        inline void gemm(type_t<TC,M0>::t C[N0], const TA *pA, const TB pB[N0][K2], const std::size_t K, const type_t<TC,M0>::t& scale) const {
            using TAC = typename type_t<bf16_t,M0*K0>::t;
            using TBC = typename type_t<bf16_t,K1*K0>::t;
            // pA[:][M0][K0]
            static_assert(N>0);
            GGML_ASSERT(K%K1 == 0);
            GGML_ASSERT(K <= K2);

            //std::cout << "gemm<"<<N<<":"<<K<<">: "<< pA <<"/"<< pB << std::endl;

            //typename type_t<TA,M0>::t A[K1]; // == TA[K1][M0][K0]
            TAC A[K1]; // == TA[K1][M0][K0]

#pragma GCC unroll N
            for(size_t j=0; j<N; j++) {
                C[j] = _mm512_setzero_ps();
            }

            for (size_t k2=0; k2<K; k2+=K1*K0) { // de 8 en 8 ...
                // chargement de A
#pragma GCC unroll K1
                for (size_t k1=0; k1<K1; ++k1) {  // [0..3]
                    A[k1] = load<TAC>(pA + k2*M0 + k1*M0*K0);
                }
#pragma GCC unroll N
                for (size_t j=0; j<N; ++j) {
                    // on charge K1 valeur de B
                    //auto B = load<TBC>(&pB[j][k2]);
#pragma GCC unroll K1
                    for (size_t k1=0; k1<K1; ++k1) {  // [0..4]
                        //auto _B = _mm512_broadcastd_2pbh(B);
                        //B = _mm_shiftl_2pbh(B);
                        auto _B = (__m512bh)_mm512_set1_ps(*(float*)(&pB[j][k2+2*k1]));
                        //auto _B = broadcast_2x(pB[j], );
                        C[j] = madd(C[j], A[k1], _B);
                    }
                }
            }

            // ecriture de C...
#pragma GCC unroll N
            for(size_t j=0; j<N; j++) {
                //if (SCALE!=Scale::NONE)    C[j] *= scale;
                if (SCALE==Scale::PER_COL) C[j] *= scale;
                static_assert(SCALE!=Scale::GLOBAL);
            }
        }

        /*
        template<typename TB, int KB>
        inline void bloc_B(bf16_t B_bloc[N0][K2], const TB* B, size_t ldb, size_t N, size_t K) {
            for (size_t j=0; j<N; j++) {
                for (size_t k=0; k<K; k+=KB) {
                    auto b = load<type_t<bf16_t,KB>::t>(B+k)+j*ldb);
                    store(&B_bloc[j][k],b);
                }
            }
        }
        */
        template<typename TB, int KB=32>
        inline void bloc_B(bf16_t B_bloc[K2], const TB* B, size_t K) const {
            for (size_t k=0; k<K; k+=KB) {
                auto b = load<typename type_t<bf16_t,KB>::t>(B+k);
                store(&B_bloc[k],b);
            }
        }

        // TODO: gerer tous les types...
        inline void sgemm_bloc(type_t<fp32_t,M0>::t C[N0], const TYPE_A* A, const bf16_t B[N0][K2], size_t N, size_t K, const type_t<fp32_t,M0>::t& scale) const {
            static_assert(N0<=16);
            switch (N) {
                case 16: gemm<16>(C, A, B, K, scale); break;
                case 15: gemm<15>(C, A, B, K, scale); break;
                case 14: gemm<14>(C, A, B, K, scale); break;
                case 13: gemm<13>(C, A, B, K, scale); break;
                case 12: gemm<12>(C, A, B, K, scale); break;
                case 11: gemm<11>(C, A, B, K, scale); break;
                case 10: gemm<10>(C, A, B, K, scale); break;
                case  9: gemm< 9>(C, A, B, K, scale); break;
                case  8: gemm< 8>(C, A, B, K, scale); break;
                case  7: gemm< 7>(C, A, B, K, scale); break;
                case  6: gemm< 6>(C, A, B, K, scale); break;
                case  5: gemm< 5>(C, A, B, K, scale); break;
                case  4: gemm< 4>(C, A, B, K, scale); break;
                case  3: gemm< 3>(C, A, B, K, scale); break;
                case  2: gemm< 2>(C, A, B, K, scale); break;
                case  1: gemm< 1>(C, A, B, K, scale); break;
                default: break;
            }
        }

        void mul_mat(const Matrice<TYPE_A,K0,M0,SCALE>& A, const Matrice<fp32_t,K0*K1>& B, Matrice<fp32_t,M0>& C) const {
            static_assert(K0==2);
            static_assert(M0==16);
            constexpr size_t N1 = 4; // 1; //4; //8;
            const auto m = C.DIM1(); // == A.DIM2()
            const auto n = C.DIM2(); // == B.DIM2()
            const auto k = A.DIM1(); // == B.DIM1()

            bf16_t B_cache[N1][N0][K2];
            typename type_t<fp32_t,M0>::t C_bloc[N0];

            // parallele region
            #pragma omp parallel private(C_bloc)
            for (size_t k2=0; k2<k; k2+=K2) {
                const auto K = std::min(k-k2,K2);
                for (size_t j2=0; j2<n; j2+=N1*N0) {
                    const auto N1_MAX=std::min(N1*N0,(n-j2));
                    // parallele for...
                    //for (size_t j1=0; j1<N1; j1+=1) {
                    //    for (size_t j0=0; j0<N0; j0+=1) {
                    //        bloc_B(B[j1][j0], B.addr(k2,j2+j1*N0+j0), K);
                    //    }
                    //}
                    #pragma omp for schedule(guided)
                    for (size_t j=0; j<N1_MAX; j++) {
                        auto j1 = j/N0;
                        auto j0 = j%N0;
                        //std::cout << "bloc_B" << std::endl;
                        bloc_B(B_cache[j1][j0], B.addr(k2,j2+j), K);
                    }
                    //#pragma omp barrier
                    #pragma omp for schedule(guided)
                    //#pragma omp parallel for private(C_bloc) shared(B_cache) schedule(guided)
                    for (size_t i1=0; i1<m; i1+=M0) {  // si on veux // sur plus de CPU il faut decouper M... et // M2/N2
                        for (size_t j1=0; j1<N1_MAX; j1+=N0) { // parcours de bloc B[N0][K1]
                            auto N = std::min(N0,N1_MAX-j1);
                            // calcul du kernel.
                            //std::cout << "sgemm_bloc <"<<N<<"/"<<N1_MAX<<":"<<K<<">:"<<j2<<":"<<j1<<":"<<i1<<":"<<k2  << std::endl;
                            sgemm_bloc(C_bloc, A.addr(k2,i1), B_cache[j1/N0], N, K, A.template get_scale_v<M0>(i1));
                            // stokage de C
                            //std::cout << "store C " <<N<<":"<<j2<<":"<<j1<<":"<<i1<<":"<<k2 << std::endl;
                            for (size_t j0=0; j0<N; j0++) {
                                if (k2==0) {
                                    store(C.addr(i1,j2+j1+j0),C_bloc[j0]);
                                } else {
                                    auto c = load<typename type_t<fp32_t,M0>::t>(C.addr(i1,j2+j1+j0));
                                    c += C_bloc[j0];
                                    store(C.addr(i1,j2+j1+j0),c);
                                }
                            }
                            //std::cout << " OK" << std::endl;
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
