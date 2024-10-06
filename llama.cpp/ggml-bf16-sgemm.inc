// work with all "natif" on A: fp32/bf16/fp8
// always compute with "dot2": fp32_t += bf16_t*bf16_t + bf16_t*bf16_t
//  => only for zen4 (5?)

// pour que ca marche il faut avoir codé:
//  - load  // A&B => __m512bh
//  - store // __m512bh => B_cache
//  - hsum  // __m512   => float

enum class ACTION {
    NONE,
    STORE,
    LOAD
};

// the lowest kernel.
// TB=TC=float !
template<size_t M, size_t N,
Scale SCALE, std::size_t MB, std::size_t KB,
ACTION ACT, bool ACC,
typename TA, typename TB, typename TC, typename TS
>
static void sgemm_bf16x32(const TA *pA, const TB *pB, TC *pC, std::size_t lda, std::size_t ldb, std::size_t ldc, std::size_t K, bf16_t *pB_, std::size_t ldb_, TS scale) {
    constexpr int K0 = 32; // 32 bf16 !
    static_assert(N>0);
    static_assert(M>0);
    // K%32 == 0!!
    // A[:,K+:lda]
    // B[:,K+:ldb]
    // C[:,ldc]
    __m512   C[M][N];
    __m512bh A[M];
#pragma GCC unroll N
    for(size_t j=0; j<N; j++) {
#pragma GCC unroll M
        for(size_t i=0; i<M; i++) {
            C[i][j] = _mm512_setzero_ps();
        }
    }

    for (std::size_t k=0; k<K; k+=K0) {
#pragma GCC unroll M
        for(size_t i=0; i<M; i++) {
            A[i] = load<__m512bh>(pA+i*lda+k);
        }
#pragma GCC unroll N
        for(size_t j=0; j<N; j++) {
            __m512bh B;
            // gestion d'un cache pour B
            if constexpr(ACT!=ACTION::LOAD) B = load<__m512bh>(pB+j*ldb+k);
            if constexpr(ACT==ACTION::LOAD) B = load<__m512bh>(pB_+j*ldb_+k);
#pragma GCC unroll M
            for(size_t i=0; i<M; i++) {
                // C[i][j] = madd(A[i], B, C[i][j]);
                C[i][j] = _mm512_dpbf16_ps(C[i][j], A[i], B);
            }
            if constexpr(ACT==ACTION::STORE) store(pB_+j*ldb_+k, B);
        }
    }

    // reduce and store C res.
#pragma GCC unroll N
    for(size_t j=0; j<N; j++) {
#pragma GCC unroll M
        for(size_t i=0; i<M; i++) {
            if constexpr(SCALE==Scale::NONE) {
                // sans scale
                if constexpr (ACC) {
                    pC[i+j*ldc] += hsum(C[i][j]);
                } else {
                    pC[i+j*ldc] = hsum(C[i][j]);
                }
            } else if constexpr(SCALE==Scale::BLOC && KB==0 && MB==0) {
                // scale globale
                if constexpr (ACC) {
                    pC[i+j*ldc] += hsum(C[i][j]) * scale;
                } else {
                    pC[i+j*ldc] = hsum(C[i][j]) * scale;
                }
            } else if constexpr(SCALE==Scale::BLOC && KB==0 && MB==1) {
                // scale par colonne
                if constexpr (ACC) {
                    pC[i+j*ldc] += hsum(C[i][j]) * scale[i];
                } else {
                    pC[i+j*ldc] = hsum(C[i][j]) * scale[i];
                }
            } else {
                // TODO les autres cas possible:
                //  - [K1,1], les autres pas possible?
            }
        }
    }
}

// dispatch for lower MxN
template<size_t M, size_t N,
Scale SCALE, std::size_t MB, std::size_t KB,
ACTION ACT, bool ACC,
typename TA, typename TB, typename TC, typename TS
>
static void sgemm_512_bloc(TA* A, TB* B, TC* C, size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc, bf16_t* B_, size_t ldb_, TS scale) {
    GGML_ASSERT(m<=M);
    GGML_ASSERT(n<=N);

    // choix du kernel:
    if ((M==m) && (N==n)) { // seul cas traité pour l'instant
        sgemm_bf16x32<M,N,SCALE,MB,KB,ACT,ACC>(A, B, C, lda, ldb, ldc, k, B_, ldb_, scale);
        return;
    }
    if constexpr (M>1) { // arret de la recursion
        if (M>m) {
            sgemm_512_bloc<M-1,N,SCALE,MB,KB,ACT,ACC>(A,B,C,m,n,k,lda,ldb,ldc, B_, ldb_, scale);
        }
    }
    if constexpr (N>1) { // arret de la recursion
        if (M==m && N>n) {
            sgemm_512_bloc<M,N-1,SCALE,MB,KB,ACT,ACC>(A,B,C,m,n,k,lda,ldb,ldc, B_, ldb_, scale);
        }
    }
}

// dispache compute
// revoir K0: K0=32/16 et K0->K1
template<size_t M1, size_t N1, size_t M0, size_t N0, size_t K0,
        Scale SCALE, size_t MB, size_t KB,
        typename TA, typename TB, typename TC
        >
static inline void sgemm_512_bloc(const Matrice<TA,32,0,0,SCALE,MB,KB>& A, const Matrice<TB,32>& B, Matrice<TC,16>& C, size_t I0, size_t J0, bf16_t* B_) {
    const size_t IN = std::min(C.DIM1(), I0+M1*M0);
    const size_t JN = std::min(C.DIM2(), J0+N1*N0);
    const auto KN = A.DIM1(); // == B.DIM1()

    if (B_) {
        // avec cache sur B
        for (size_t k=0; k<KN; k+=K0) {
            const auto _K = std::min(K0,KN-k);
            for (size_t j=J0; j<JN; j+=N0) {
                const auto _N = std::min(N0,JN-j);
                if (k==0) {
                    sgemm_512_bloc<M0,N0,SCALE,MB,KB,ACTION::STORE,false>(A.addr(0,I0),B.addr(0,j),C.addr(I0,j),std::min(M0,IN-I0),_N,_K, A.LD(),B.LD(),C.LD(), B_, K0, A.get_scale(I0));
                } else {
                    sgemm_512_bloc<M0,N0,SCALE,MB,KB,ACTION::STORE,true>(A.addr(k,I0),B.addr(k,j),C.addr(I0,j),std::min(M0,IN-I0),_N,_K, A.LD(),B.LD(),C.LD(), B_, K0, A.get_scale(I0));
                }
                if (I0+M0<IN)
                    for (size_t i=I0+M0; i<IN; i+=M0) {
                        const auto _M = std::min(M0,IN-i);
                        if (k==0) {
                            sgemm_512_bloc<M0,N0,SCALE,MB,KB,ACTION::LOAD,false>(A.addr(0,i),B.addr(0,j),C.addr(i,j),_M,_N,_K, A.LD(),B.LD(),C.LD(), B_, K0, A.get_scale(i));
                        } else {
                            sgemm_512_bloc<M0,N0,SCALE,MB,KB,ACTION::LOAD,true>(A.addr(k,i),B.addr(k,j),C.addr(i,j),_M,_N,_K, A.LD(),B.LD(),C.LD(), B_, K0, A.get_scale(i));
                        }
                    }
            }
        }
    } else {
        // tjs usage de B (ie sans cache!)
        for (size_t k=0; k<KN; k+=K0) {
            const auto _K = std::min(K0,KN-k);
            for (size_t j=J0; j<JN; j+=N0) {
                const auto _N = std::min(N0,JN-j);
                for (size_t i=I0; i<IN; i+=M0) {
                    const auto _M = std::min(M0,IN-i);
                    if (k==0) {
                        sgemm_512_bloc<M0,N0,SCALE,MB,KB,ACTION::NONE,false>(A.addr(0,i),B.addr(0,j),C.addr(i,j),_M,_N,_K, A.LD(),B.LD(),C.LD(), B_, K0, A.get_scale(i));
                    } else {
                        sgemm_512_bloc<M0,N0,SCALE,MB,KB,ACTION::NONE,true>(A.addr(k,i),B.addr(k,j),C.addr(i,j),_M,_N,_K, A.LD(),B.LD(),C.LD(), B_, K0, A.get_scale(i));
                    }
                }
            }
        }
    }
}

template<typename TYPE_A, Scale SCALE, size_t MB, size_t KB>
class ggml_bf16_op_matmul : public ggml::backend::bf16::op {
public:
    using tensorA_t = ggml::backend::bf16::tensor<TYPE_A,32,0,0,SCALE,MB,KB>;
    using tensorB_t = ggml::backend::bf16::tensor<fp32_t,32>;
    using tensorC_t = ggml::backend::bf16::tensor<fp32_t,16>;
    using matA_t = tensorA_t::matrice_t;
    using matB_t = tensorB_t::matrice_t;
    using matC_t = tensorC_t::matrice_t;

    // TODO: definir le type en fct du CPU AVX2/AVX512/BF16/FP32...
    //  - pour l'instant tjs AVX512/BF16 !
    virtual bool C_is_allowed(const struct ggml_tensor *C) {
        return tensorC_t::type()->is_allowed(C) == ggml::backend::bf16::Tensor_t::COMPAT::NATIVE;
    }
    virtual bool B_is_allowed(const struct ggml_tensor *B) {
        return tensorB_t::type()->is_allowed(B) == ggml::backend::bf16::Tensor_t::COMPAT::NATIVE;
    }
    bool A_is_allowed(const struct ggml_tensor *A) override {
        auto a = ggml::backend::bf16::Tensor_t::GET(A);
        if(!a && A->view_src) {
            // une vue => pas "encore" supporté
            return false;
        }
        GGML_ASSERT(a!=nullptr);
        // on sais deja:
        if (a == tensorA_t::type()) return true;
        return false;
    }

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

                const matA_t A(src0,i02,i03);
                const matB_t B(src1,i12,i13);
                matC_t       C(dst,i12,i13);
                mul_mat(A, B, C);
            }}
#ifdef DO_TIMING
        auto dt = time.end();
        std::cout << " > " <<op->op<<"("<<log_srcs(op)<<" => "<<op<<"): "<< op->name << " => "
                << dt*1000000 << " us / "
                << (1e-9*2*src1->ne[3]*src1->ne[2]*op->ne[0]*op->ne[1]*src1->ne[0])/dt << " GFlops/s"
                << std::endl;
#endif
    }
    //  - bf16+fp32=fp32
    virtual void mul_mat(const matA_t& A, const matB_t& B, matC_t& C) const = 0;
};

// K1 = 2560 5120 2048 4096 ...
// compute for tg and small pp
template<typename TYPE_A, Scale SCALE, size_t MB, size_t KB,
size_t M0, size_t N0, size_t M1=4, size_t K1=5120
>
class ggml_bf16_op_matmul_tg : public ggml_bf16_op_matmul<TYPE_A,SCALE,MB,KB> {
    using parent = ggml_bf16_op_matmul<TYPE_A,SCALE,MB,KB>;
public:
    bool B_is_allowed(const struct ggml_tensor *B) override {
        // limit for tg and small pp (<=N0)
        if (B->ne[1]>=N0) return false;
        return parent::B_is_allowed(B);
    }
    void mul_mat(const parent::matA_t& A, const parent::matB_t& B, parent::matC_t& C) const override {
        const auto m = C.DIM1(); // == A.DIM2()
        const auto n = C.DIM2(); // == B.DIM2()
        const auto k = A.DIM1(); // == B.DIM1()
        GGML_ASSERT(A.LD()>=k);
        GGML_ASSERT(B.LD()>=k);
        GGML_ASSERT(C.LD()>=m);
        bf16_t B_cache[N0*K1];

#pragma omp parallel for private(B_cache) schedule(guided) num_threads(4)
        //#pragma omp parallel for schedule(guided)
        for (size_t i=0; i<m; i+=M1*M0) {
            //sgemm_512_bloc<M1,1,M0,N0,K1>(A, B, C, i, 0, nullptr);
            sgemm_512_bloc<M1,1,M0,N0,K1>(A, B, C, i, 0, B_cache);
        }
    }
};

// dispache for all pp size
template<typename TYPE_A, Scale SCALE, size_t MB, size_t KB,
size_t M0, size_t N0, size_t M1=8, size_t N1=4, size_t K1=2560
>
class ggml_bf16_op_matmul_pp : public ggml_bf16_op_matmul<TYPE_A,SCALE,MB,KB> {
    using parent = ggml_bf16_op_matmul<TYPE_A,SCALE,MB,KB>;
public:
    void mul_mat(const parent::matA_t& A, const parent::matB_t& B, parent::matC_t& C) const override {
        const auto m = C.DIM1(); // == A.DIM2()
        const auto n = C.DIM2(); // == B.DIM2()
        const auto k = A.DIM1(); // == B.DIM1()
        GGML_ASSERT(A.LD()>=k);
        GGML_ASSERT(B.LD()>=k);
        GGML_ASSERT(C.LD()>=m);

        // la taille des plus grand blocs.
        bf16_t B_cache[N0*K1];

        // schedule(dynamique)
#pragma omp parallel for collapse(2) private(B_cache) schedule(guided)
        for (size_t i=0; i<m; i+=M1*M0) {
            for (size_t j=0; j<n; j+=N1*N0) {
                //sgemm_512_bloc<M1,N1,M0,N0,K1>(A, B, C, i, j, nullptr);
                sgemm_512_bloc<M1,N1,M0,N0,K1>(A, B, C, i, j, B_cache);
            }
        }
    }
};
