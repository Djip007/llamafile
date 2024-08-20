// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi

#ifdef __x86_64__

#include "llama_matmul.h"

#include <immintrin.h>
#include <stdalign.h>
#include <stdatomic.h>
#include <string.h>
#include <unistd.h>

#define MR 16
#define NR 6

// Consider fine-tuning the following parameters for your CPU
#define MC MR *NTHREADS * 4
#define NC NR *NTHREADS * 32
#define KC 1000

#define ALIGNED __attribute__((__aligned__(64)))

#define min(x, y) ((x) < (y) ? (x) : (y))

#define _mm256_loadu_hs(u16ptr) \
    _mm256_castsi256_ps( \
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(u16ptr))), 16))

static void syncthreads(int ith) {
    static atomic_uint count;
    static struct {
        alignas(64) atomic_uint i;
    } ph[NTHREADS];
    int phase = atomic_load_explicit(&ph[ith].i, memory_order_relaxed);
    if (atomic_fetch_add_explicit(&count, 1, memory_order_acq_rel) + 1 == NTHREADS) {
        atomic_store_explicit(&count, 0, memory_order_relaxed);
        for (int i = 0; i < NTHREADS; ++i)
            atomic_store_explicit(&ph[i].i, phase + 1, memory_order_relaxed);
        atomic_thread_fence(memory_order_release);
    } else {
        for (;;)
            if (atomic_load_explicit(&ph[ith].i, memory_order_relaxed) != phase)
                break;
        atomic_thread_fence(memory_order_acquire);
    }
}

static float blockB_packed[NC * KC] ALIGNED;
static uint16_t blockA_packed[MC * KC] ALIGNED;

static int8_t mask_32[32] ALIGNED = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};

static float from_brain(uint16_t h) {
    union {
        float f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h << 16;
    return u.f;
}

static void pack_panelB(const uint16_t *B, float *blockB_packed, const int nr, const int kc,
                        const long ldb) {
    for (int p = 0; p < kc; p++) {
        for (int j = 0; j < nr; j++) {
            *blockB_packed++ = from_brain(B[j * ldb + p]);
        }
        for (int j = nr; j < NR; j++) {
            *blockB_packed++ = 0;
        }
    }
}

static dontinline void pack_blockB(const uint16_t *B, float *blockB_packed, const int nc,
                                   const int kc, const long ldb, const int ith) {
    for (int j = ith * NR; j < nc; j += NR * NTHREADS) {
        const int nr = min(NR, nc - j);
        pack_panelB(&B[j * ldb], &blockB_packed[j * kc], nr, kc, ldb);
    }
}

static void pack_panelA(const uint16_t *A, uint16_t *blockA_packed, const int mr, const int kc,
                        const long lda) {
    for (int p = 0; p < kc; p++) {
        for (int i = 0; i < mr; i++) {
            *blockA_packed++ = A[i * lda + p];
        }
        for (int i = mr; i < MR; i++) {
            *blockA_packed++ = 0;
        }
    }
}

static dontinline void pack_blockA(const uint16_t *A, uint16_t *blockA_packed, const int mc,
                                   const int kc, const long lda, const int ith) {
    for (int i = ith * MR; i < mc; i += MR * NTHREADS) {
        const int mr = min(MR, mc - i);
        pack_panelA(&A[i * lda], &blockA_packed[i * kc], mr, kc, lda);
    }
}

static dontinline void llama_matmul_kernel_16x6_avx2_bf16(uint16_t *blockA_packed,
                                                          float *blockB_packed, float *C,
                                                          const int m, const int n, const int k,
                                                          const long ldc) {
    __m256 C_buffer[2][6];
    __m256 b_packFloat8;
    __m256 a0_packFloat8;
    __m256 a1_packFloat8;
    __m256i packed_masks[2] = {};
    if (m != 16) {
        packed_masks[0] = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask_32[16 - m]));
        packed_masks[1] = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask_32[16 - m + 8]));
        for (int j = 0; j < n; j++) {
            C_buffer[0][j] = _mm256_maskload_ps(&C[j * ldc], packed_masks[0]);
            C_buffer[1][j] = _mm256_maskload_ps(&C[j * ldc + 8], packed_masks[1]);
        }
    } else {
        for (int j = 0; j < n; j++) {
            C_buffer[0][j] = _mm256_loadu_ps(&C[j * ldc]);
            C_buffer[1][j] = _mm256_loadu_ps(&C[j * ldc + 8]);
        }
    }
    for (int p = 0; p < k; p++) {
        a0_packFloat8 = _mm256_loadu_hs(blockA_packed);
        a1_packFloat8 = _mm256_loadu_hs(blockA_packed + 8);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
        C_buffer[0][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][0]);
        C_buffer[1][0] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][0]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
        C_buffer[0][1] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][1]);
        C_buffer[1][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][1]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
        C_buffer[0][2] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][2]);
        C_buffer[1][2] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][2]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 3);
        C_buffer[0][3] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][3]);
        C_buffer[1][3] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][3]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 4);
        C_buffer[0][4] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][4]);
        C_buffer[1][4] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][4]);

        b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 5);
        C_buffer[0][5] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][5]);
        C_buffer[1][5] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][5]);

        blockA_packed += 16;
        blockB_packed += 6;
    }
    if (m != 16) {
        for (int j = 0; j < n; j++) {
            _mm256_maskstore_ps(&C[j * ldc], packed_masks[0], C_buffer[0][j]);
            _mm256_maskstore_ps(&C[j * ldc + 8], packed_masks[1], C_buffer[1][j]);
        }
    } else {
        for (int j = 0; j < n; j++) {
            _mm256_storeu_ps(&C[j * ldc], C_buffer[0][j]);
            _mm256_storeu_ps(&C[j * ldc + 8], C_buffer[1][j]);
        }
    }
}

static void clear_tile(float *C, const int m, const int n, const long ldc) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            C[j * ldc + i] = 0;
        }
    }
}

void llama_matmul_avx2_bf16(const uint16_t *A, const uint16_t *B, float *C, const int M,
                            const int N, const int K, const long lda, const long ldb,
                            const long ldc, const int ith) {
    for (int j = 0; j < N; j += NC) {
        const int nc = min(NC, N - j);
        for (int i = 0; i < M; i += MC) {
            const int mc = min(MC, M - i);
            for (int jr = ith * NR; jr < nc; jr += NR * NTHREADS) {
                const int nr = min(NR, nc - jr);
                for (int ir = 0; ir < mc; ir += MR) {
                    const int mr = min(MR, mc - ir);
                    clear_tile(&C[(j + jr) * M + (i + ir)], mr, nr, ldc);
                }
            }
        }
    }
    for (int j = 0; j < N; j += NC) {
        const int nc = min(NC, N - j);
        for (int p = 0; p < K; p += KC) {
            const int kc = min(KC, K - p);
            pack_blockB(&B[j * K + p], blockB_packed, nc, kc, ldb, ith);
            for (int i = 0; i < M; i += MC) {
                const int mc = min(MC, M - i);
                pack_blockA(&A[i * K + p], blockA_packed, mc, kc, lda, ith);
                syncthreads(ith);
                for (int jr = ith * NR; jr < nc; jr += NR * NTHREADS) {
                    const int nr = min(NR, nc - jr);
                    for (int ir = 0; ir < mc; ir += MR) {
                        const int mr = min(MR, mc - ir);
                        llama_matmul_kernel_16x6_avx2_bf16(
                            &blockA_packed[ir * kc], &blockB_packed[jr * kc],
                            &C[(j + jr) * M + (i + ir)], mr, nr, kc, ldc);
                    }
                }
                syncthreads(ith);
            }
        }
    }
}

#endif // __x86_64__
