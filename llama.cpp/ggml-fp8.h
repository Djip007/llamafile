// this is more a .inc.
#ifdef  __cplusplus
extern "C" {
#endif

    // Note: types are define in ggml-common.h

    GGML_API void ggml_e5m2_to_fp32_row(const ggml_e5m2_t * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
    GGML_API void ggml_fp32_to_e5m2_row(const float * GGML_RESTRICT x, ggml_e5m2_t * GGML_RESTRICT y, int64_t k);
    GGML_API void ggml_fp32_to_e5m2_row_ref(const float * GGML_RESTRICT x, ggml_e5m2_t * GGML_RESTRICT y, int64_t k);

    GGML_API void ggml_e4m3_to_fp32_row(const ggml_e4m3_t * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
    GGML_API void ggml_fp32_to_e4m3_row(const float * GGML_RESTRICT x, ggml_e4m3_t * GGML_RESTRICT y, int64_t k);
    GGML_API void ggml_fp32_to_e4m3_row_ref(const float * GGML_RESTRICT x, ggml_e4m3_t * GGML_RESTRICT y, int64_t k);

    GGML_API void dequantize_row_e4m3_q(const block_e4m3_q * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
    GGML_API void quantize_row_e4m3_q(const float * GGML_RESTRICT x, block_e4m3_q * GGML_RESTRICT y, int64_t k);
    GGML_API void quantize_row_e4m3_q_ref(const float * GGML_RESTRICT x, block_e4m3_q * GGML_RESTRICT y, int64_t k);

    GGML_API void dequantize_row_e3m4_q(const block_e3m4_q * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
    GGML_API void quantize_row_e3m4_q(const float * GGML_RESTRICT x, block_e3m4_q * GGML_RESTRICT y, int64_t k);
    GGML_API void quantize_row_e3m4_q_ref(const float * GGML_RESTRICT x, block_e3m4_q * GGML_RESTRICT y, int64_t k);

    // TODO: the best depend on the CPU fp32 / bf16 / fp16
#define GGML_FP8_VECT_DOT_TYPE GGML_TYPE_F32
    GGML_API void ggml_vec_dot_e5m2(int n, float * GGML_RESTRICT s, size_t bs, const ggml_e5m2_t * GGML_RESTRICT vx, size_t bx, const float * GGML_RESTRICT vy, size_t by, int nrc);
    GGML_API void ggml_vec_dot_e4m3(int n, float * GGML_RESTRICT s, size_t bs, const ggml_e4m3_t * GGML_RESTRICT vx, size_t bx, const float * GGML_RESTRICT vy, size_t by, int nrc);
    GGML_API void ggml_vec_dot_e4m3_q(int n, float * GGML_RESTRICT s, size_t bs, const block_e4m3_q * GGML_RESTRICT vx, size_t bx, const float * GGML_RESTRICT vy, size_t by, int nrc);
    GGML_API void ggml_vec_dot_e3m4_q(int n, float * GGML_RESTRICT s, size_t bs, const block_e3m4_q * GGML_RESTRICT vx, size_t bx, const float * GGML_RESTRICT vy, size_t by, int nrc);

#ifdef  __cplusplus
}
#endif
