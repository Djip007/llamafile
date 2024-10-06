// qq elements: on cherche ca:
//  - A => en L2
//  - B => en L1
// => il fait gerer des K1
// => parcourir M2 bloc
// => parcourir tous les B
// => passer a la suite...
//   - K0 : dot2       => 2
//   - M0 : simd       => 16
//   - N0 : registres  => 16?
//   - K1 : B[K1,N0]    en L1  => 32k: /16 = 1024 max
//   - M1 : A[K1,M0*M1] en L2  => 1M  :(bf16|fp8) / (16*2*1024)|(16*1*1024) = 32 | 64 max ...
//   - N1 : B[K1,N0*N1] en L3? => 12M : bf16 / 16*2*1024 = >256  (N0*N1=4096)
//   - M  : tous les M   => dispache par thread
//   - N  : tous les N   => dispache par SOCKET / L3 cache / ... pas gagné ...
//   - K  : tous les K   => si on veux l'utiliser il faudrait faire des reductions...

namespace ggml::bf16::op_matmul {
    // TODO gerer les autres types...
    //  - COMPUTE_TYPE ACC_TYPE ? FP32 += FP16*FP16 ...
    //  - C_TYPE / B_TYPE ... pour l'instant tjs fp32 ?
    template<typename TYPE_A,
    std::size_t M0=16, std::size_t N0=16, std::size_t K0=2, std::size_t K1=1024,
    Scale SCALE=Scale::NONE, std::size_t MB=0, std::size_t KB=0>
    class bf16_2x16: public ggml::backend::bf16::op {
    public:
        using type_A = TYPE_A;
        using type_B = fp32_t;
        using type_C = fp32_t;
        //using matA_t = Matrice<type_A,K0,M0,K1,SCALE,MB,KB>;
        //using matB_t = Matrice<type_B,K1>;
        //using matC_t = Matrice<type_C,M0>;
        using tensorA_t = ggml::backend::bf16::tensor<type_A,K0,M0,K1,SCALE,MB,KB>;
        using tensorB_t = ggml::backend::bf16::tensor<type_B,K1>;
        using tensorC_t = ggml::backend::bf16::tensor<type_C,M0>;

        using matA_t = tensorA_t::matrice_t;
        using matB_t = tensorB_t::matrice_t;
        using matC_t = tensorC_t::matrice_t;

        // les types pour les calculs
        using type_AB = bf16_t; // le type utilisé pour les calculs
        static constexpr int VS=32;  // ca demande de ce que support le CPU et du type_AB == K0*M0 ???
        // autre cas:
        // bf16_32(AVX512BF) / float_16 (AVX512) / float_8 (AVX2)
        // NEON?

        bf16_2x16() {
            static_assert(SCALE==Scale::NONE || SCALE==Scale::BLOC); // en attandant de supprimer les autres cas!
            static_assert(K1%K0==0);
            static_assert(K0==2);  // pour l'instant pas d'autre cas pour calcul en bf16 / 1 si calcul en fp32 ...
            static_assert(M0==16); // 16xFP32 => 1 AVX512 / 8xFP32 => 1 AVX256 ...
        }
        void exec(struct ggml_tensor *op) const override {
#ifdef DO_TIMING
            mesure time; time.start();
#endif
            // normalement ici les type sont deja controlé
            const matA_t A(op->src[0]);
            const matB_t B(op->src[1]);
            matC_t C(op);

            //std::cout << "#> bf16_2x16::exec<"<<K0<<","<<M0<<","<<N0<<","<<K1<<">:"<<op->name<<"("<<op->src[0]->name<<","<<op->src[1]->name<<")"<< std::endl;
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
            return tensorB_t::type()->is_allowed(B) == ggml::backend::bf16::Tensor_t::COMPAT::NATIVE;
        }
        bool C_is_allowed(const struct ggml_tensor *C) override {
            return tensorC_t::type()->is_allowed(C) == ggml::backend::bf16::Tensor_t::COMPAT::NATIVE;
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
            if (a == tensorA_t::type()) return true;
            return false;
        }

    private:

        // les 2 doivent etre "synchro"
        static inline auto get_scale(const matA_t& A, std::size_t k0, std::size_t i0 ) {
            if constexpr(SCALE==Scale::NONE) return (float)1; // de toute facon il est pas utilisé!
            if constexpr(SCALE==Scale::BLOC) return A.addr_scale(k0,i0);
        }
        template<typename SCALE_T>
        static inline auto load_scale(const SCALE_T scale, std::size_t k) {
            // if constexpr(SCALE==Scale::NONE) return 0; // ne doit pas etre utilisé!
            if constexpr(SCALE==Scale::BLOC) {
                // OK on a un const float* en entrée a priorie...
                // Il y a 2 cas: broadcast ou vecteur...
                // - broadcast:
                if constexpr(KB==0 && MB==0) {
                    // GLOBAL : KB=MB=0  => broadcast
                    return _mm512_set1_ps(*scale);
                } else if constexpr(KB>=K0 && MB>=M0 && KB%K0==0 && MB%M0==0) {
                    // BLOC   : MB>M1 && KB>0
                    return _mm512_set1_ps(scale[k/KB]);
                } else if constexpr( MB==1 && (KB==0 || (KB>=K1 && KB%K1==0)) ) {
                    // COLONNE globale ou COLONNE d'un bloc => on charge les M0 donnée
                    return load<typename type_t<type_C,M0>::t>(scale);
                } else {
                    static_assert(false,"Pas encore supporté... est-ce possible");
                }
            } else {
                static_assert(false,"On ne devrais jamais l'utiliser");
            }
        }
        static inline auto load_B(const type_AB* pB) {
            // TODO: il faut broadcaster pB[K0] sur le vecteur de calcule
            // on charge K1 valeur de B
            //auto B = load<TBC>(&pB[j][k2]);
            //auto B = broadcast_2x(pB[j], );
            //auto B = broadcast<bf16_t,8,K0>(&pB[j][k2]);
            return (__m512bh)_mm512_set1_ps(*(float*)(pB));
        }
        static inline auto load_A(const type_A* pA) {
            return load<typename type_t<type_AB,M0*K0>::t>(pA);
        }
        static inline void init_C(type_t<type_C,M0>::t& C) {
            // TODO: mise a 0 generique
            C = _mm512_setzero_ps();
        }
        static inline type_t<type_C,M0>::t load_C(type_C* pC) {
            return load<typename type_t<type_C,M0>::t>(pC);
        }
        // le bloc de bas niveau
        // TODO ajouter in gemv (mat/vect): N=N0=1 !!

        // TA pour l'instant bf16 / fp8_...
        template<size_t N, typename SCALE_T, size_t UN=(KB==0?16:(KB/K0))>
        inline void gemm(type_C pC[N0][M0], const type_A *pA, const type_AB pB[N0][K1], const SCALE_T scale) const {
            // pA[:][M0][K0]
            //using TAC = typename type_t<type_AB,M0*K0>::t;
            static_assert(N>0);
            typename type_t<type_C,M0>::t C[N0];

            if constexpr(SCALE==Scale::NONE) {
#pragma GCC unroll N
                for(size_t j=0; j<N; j++) { init_C(C[j]); }
            }

            // si KB == 0 on les prand tous... ici c'est pareil que en prandre K1
            constexpr size_t KP = KB>0?KB:K1;
            for (size_t k1=0; k1<K1; k1+=KP) {
                if constexpr(SCALE==Scale::BLOC) {
#pragma GCC unroll N
                    for(size_t j=0; j<N; j++) { init_C(C[j]); }
                }
#pragma GCC unroll 8
                // le bloc sur lequel il faut appliquer le scale
                for (size_t k0=0; k0<KP; k0+=K0) {
                    // chargement de A
                    auto A = load_A(pA + (k1+k0)*M0); // == ~TA[M0][K0]
#pragma GCC unroll N
                    for (size_t j=0; j<N; ++j) {
                        auto B = load_B(&pB[j][k1+k0]);
                        C[j] = madd(C[j], A, B);
                    }
                }
                // facteur de correction:
                if constexpr(SCALE==Scale::BLOC) {
                    auto _scale = load_scale(scale, k1);
#pragma GCC unroll N
                    for(size_t j=0; j<N; j++) {
                        auto c = load_C(pC[j]);
                        c = madd(c,C[j],_scale);
                        // C[j] *= _scale;
                        store(pC[j], c);
                    }
                }
            }
            if constexpr(SCALE==Scale::NONE) {
#pragma GCC unroll N
                for(size_t j=0; j<N; j++) {
                    auto c = load_C(pC[j]);
                    c += C[j];
                    store(pC[j], c);
                }
            }
        }

        inline void bloc_B(type_AB B_bloc[K1], const type_B* B) const {
            // KB la taille du vecteur de calcul : celuis de l'AVX utilisé
            // le "mieux" serait encore de reformater B_bloc[K1/K0][N][K0] ...
            // #pragma GCC ivdep => unroll SIMD ...
            for (size_t k=0; k<K1; k+=VS) {
                auto b = load<typename type_t<type_AB,VS>::t>(B+k);
                store(&B_bloc[k],b);
            }
        }

        template<typename SCALE_T>
        inline void sgemm_bloc(type_C C[N0][M0], const type_A* A, const type_AB B[N0][K1], size_t N, const SCALE_T scale) const {
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

        void mul_mat(const matA_t& A, const matB_t& B, matC_t& C) const {
            constexpr size_t N1 = 8; // 1; //4; //8;
            constexpr size_t M1 = 8; // 1; //4; //8;

            const auto m = C.DIM1(); // == A.DIM2()
            const auto n = C.DIM2(); // == B.DIM2()
            const auto k = A.DIM1(); // == B.DIM1()
            GGML_ASSERT(m == A.DIM2());
            GGML_ASSERT(n == B.DIM2());
            GGML_ASSERT(k == B.DIM1());
            // GGML_ASSERT(n%(N0*N1) == 0);
            GGML_ASSERT(k%(K0) == 0);
            GGML_ASSERT(K1==0 || k%(K1) == 0);
            GGML_ASSERT(KB==0 || k%(KB) == 0);
            GGML_ASSERT(m%(M0*M1) == 0);
            GGML_ASSERT(MB==0 || m%(MB) == 0);

            // cas pour N=1 ... utilisé pour le "tg"
            // + cas pour N<16 ???
            if (n==1) {
                type_C C_bloc[N0][M0];
                type_AB B_cache[N0][K1];

#pragma omp parallel private(C_bloc) num_threads(4)
                for (size_t k2=0; k2<k; k2+=K1) {
#pragma omp master
                    { bloc_B(B_cache[0], B.addr(k2,0)); }

#pragma omp for schedule(dynamic, 8)
                    for (size_t i1=0; i1<m; i1+=M0) {
                        if (k2==0) {
#pragma omp simd
                            for (int i0=0; i0<N0; i0++) C_bloc[0][i0]=0;
                        } else {
#pragma omp simd
                            for (int i0=0; i0<N0; i0++) C_bloc[0][i0]=C.addr(i1,0)[i0];
                        }
                        // calcul du kernel.
                        gemm<1>(C_bloc, A.addr(k2,i1), B_cache, get_scale(A,k2,i1));
                        // stokage de C
#pragma omp simd
                        for (int i0=0; i0<M0; i0++) C.addr(i1,0)[i0]=C_bloc[0][i0];
                    }
                }
            } else {
                type_C C_bloc[N0][M0];
                type_AB B_cache[N1][N0][K1];

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
                            bloc_B(B_cache[j1][j0], B.addr(k2,j2+j));
                        }
                        //#pragma omp for schedule(dynamic, 8)
                        //#pragma omp for schedule(static, 8)
#pragma omp for schedule(guided)
                        //#pragma omp parallel for private(C_bloc) shared(B_cache) schedule(guided)
                        for (size_t i2=0; i2<m; i2+=M1*M0) { // parcours de bloc A[M0][K1]
                            for (size_t j1=0; j1<N1_MAX; j1+=N0) { // parcours de bloc B[N0][K1]
                                auto N = std::min(N0,N1_MAX-j1);
                                for (size_t i1=0; i1<M1*M0; i1+=M0) {
                                    // recuperation de C
                                    if (k2==0) {
                                        for (int j0=0; j0<N; j0++)
                                            for (int i0=0; i0<M0; i0++) C_bloc[j0][i0]=0;
                                    } else {
                                        for (int j0=0; j0<N; j0++) {
                                            auto pC = C.addr(i2+i1,j2+j1+j0);
                                            for (int i0=0; i0<M0; i0++) C_bloc[j0][i0]=pC[i0];
                                        }
                                    }
                                    // calcul du kernel.
                                    sgemm_bloc(C_bloc, A.addr(k2,i2+i1), B_cache[j1/N0], N, get_scale(A,k2,i2+i1));
                                    // stokage de C
                                    for (int j0=0; j0<N; j0++) {
                                        auto pC = C.addr(i2+i1,j2+j1+j0);
                                        for (int i0=0; i0<M0; i0++) pC[i0]=C_bloc[j0][i0];
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

//#pragma omp parallel private(C_bloc) shared(A,B,C,B_cache) // num_threads(4)
//#pragma omp single
//#pragma omp master
//#pragma omp for schedule(dynamic, 16)
//#pragma omp parallel for private(C_bloc) shared(A,B,C,B_cache,k2,m,n,k)
