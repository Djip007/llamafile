// test avec bf16 (zen4 [+ rdna3 apu]
// TODO le rendre "dynamique" (voir cuda.c / metal.c)

#include "llamafile.h"
#include <cosmo.h>

// ??? pas utils les 2 methodes d'enregistrement sont auto-suffisante!
bool llamafile_has_apu_amd(void) {
#ifdef __x86_64__
    return (X86_HAVE(FMA) && X86_HAVE(F16C) && X86_HAVE(AVX2) && X86_HAVE(AVX512F) && X86_HAVE(AVX512BW) && X86_HAVE(AVX512DQ) && X86_HAVE(AVX512VL) && X86_HAVE(AVX512_BF16));
#else
    return false;
#endif
}

static void import_amd_apu(void) {
    if (false) {
        tinyprint(2, "fatal error: support for --gpu ", llamafile_describe_gpu(),
                  " was explicitly requested, but it wasn't available\n", NULL);
        exit(1);
    }
}

bool llamafile_use_amd_apu(void) {
    if (FLAG_gpu == LLAMAFILE_APU_AMD) {
        //cosmo_once(&ggml_amd_apu.once, import_amd_apu);
        //return ggml_amd_apu.supported;
        return true;
    }
    return false;
}

// ce qu'il faut fournir pour le rendre dynamique:
// supprimer le static et coder cette fonction... ???
/*
bool ggml_backend_bf16_enable() {
    if (getenv("GGML_USE_BACKEND_BF16") == nullptr) return nullptr;
    // voir ce qui peut conditioner le backend...
    //  cas llamafile ???
    return (X86_HAVE(FMA) && X86_HAVE(F16C) && X86_HAVE(AVX2) && X86_HAVE(AVX512F) && X86_HAVE(AVX512BW) && X86_HAVE(AVX512DQ) && X86_HAVE(AVX512VL) && X86_HAVE(AVX512_BF16));
}
*/
/*
ggml_backend_t ggml_backend_bf16_init(void) {
    // si chargé??
    if (llamafile_use_amd_apu()) {
        return ggml_apu_amd.init();
    }
    return nullptr
}
ggml_backend_buffer_type_t ggml_backend_bf16_buffer_type(void) {
    if (llamafile_use_amd_apu()) {
        return ggml_apu_amd.buffer_type();
    }
    return nullptr
}
*/
