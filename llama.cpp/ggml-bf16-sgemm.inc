// permet de gerer les format "natif"
//   ou presque ...
// et calcule avec "dot2": fp32_t += bf16_t*bf16_t + bf16_t*bf16_t

// pour que ca marche il fautavoir codé:
//  - load  // A&B => __m512bh
//  - store // __m512bh => B_cache
//  - hsum  // __m512   => float

enum class ACTION {
    NONE,
    STORE,
    LOAD
};

// TODO: ajouter un facteur de correction pour C...
template<size_t M, size_t N, ACTION ACT=ACTION::NONE, bool ACC=false, typename TA, typename TB> //, typename TC>
static void sgemm_bf16x32(const TA *pA, const TB *pB, float *pC, std::size_t lda, std::size_t ldb, std::size_t ldc, std::size_t K, bf16_t *pB_=nullptr, std::size_t ldb_=0) {
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
                // C[i][j] = madd(A[i], B, C[i][j]);
                C[i][j] = _mm512_dpbf16_ps(C[i][j], A[i], B);
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
        sgemm_bf16x32<M,N,ACT,ACC>(A, B, C, lda, ldb, ldc, k, B_, ldb_);
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
