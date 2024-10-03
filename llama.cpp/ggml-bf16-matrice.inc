//------------------------------------------------------------------------------------
// gestion de Matrice (en attandant mieux)
// @ mettre au propre: K<>m M<>n
// => TODO gerer des bloc de taille [K1,M1,N1] ??? ...

enum class Scale {
    // TODO: ne garder que NONE et BLOC les autres sont "deductible"
    // il faut ajouter les KS/MS dans les templates
    NONE,
    GLOBAL,
    PER_COL,
    PER_BLOC,  // la taille des bloc de K1.
};

namespace std {
constexpr std::string to_string(Scale s) {
    if (s==Scale::NONE) return "NONE";
    if (s==Scale::GLOBAL) return "GLOBAL";
    if (s==Scale::PER_COL) return "PER_COL";
    if (s==Scale::PER_BLOC) return "PER_BLOC";
    return "UNKOW";
}
}

// stockage @ gerer:
//  A[ 1][ 1][ 0][K][M]    => les natifs en type simple
//  A[K0][ 1][ 0][K/K0][M]
//  A[K0][M0][ 0][K/K0][M/M0]
//  A[K0][M0][K1][M/M0][K/K0*K1]
// TODO: cas ou  K%(K0*K1)!=0  => le dernier bloc K == [K0][M0][K%(K0*K1)][M/M0] ...
// TODO: revoir la gestion des scale:
// ajouter KS/MS  0/0==global,
// K0=32,M0=0,K1=0
// K0=2,M0=16,K1=1024 / PER_COL
template<typename T, std::size_t K0=1, std::size_t M0=0, std::size_t K1=0, Scale SCALE=Scale::NONE>
class Matrice {
    static constexpr bool NATIF = (M0==0)&&(K1==0); // && SCALE==Scale::NONE ???
public:
    // qq-aides:
    //_l(
    static inline std::size_t next_bloc(std::size_t l, std::size_t k, std::size_t m) {
        return NATIF?l:(K1==0?k*M0:m*K1);
    }
    static inline std::size_t scale_size(const struct ggml_tensor * t) {
        if constexpr(SCALE == Scale::NONE)    return 0;
        if constexpr(SCALE == Scale::GLOBAL)  return TENSOR_ALIGNMENT;
        if constexpr(SCALE == Scale::PER_COL) return sizeof(float)*t->ne[1];
    }

    // inline Matrice(struct ggml_tensor * t):
    static inline std::size_t size(const struct ggml_tensor * t) {
        if constexpr (NATIF) {
            return ggml_nbytes(t);
        } else {
            // il sera convertit / paqué ...etc
            GGML_ASSERT(t->ne[2]==1);
            GGML_ASSERT(t->ne[3]==1);
            auto size = sizeof(T)*t->ne[0]*t->ne[1]; //*tensor->ne[2]*tensor->ne[3];
            return size+scale_size(t);
        }
    }

    static inline float* scale_adr(const void * data) {
        if constexpr(SCALE == Scale::NONE) {
            return nullptr;
        } else {
            return (float*)data;
        }
    }

    static inline T* data_adr(const void * addr, std::size_t k, std::size_t m, std::size_t shift = 0) {
        static_assert(SCALE != Scale::PER_BLOC || K1==0); // si BLOC alors il faut avoir K1
        if constexpr(SCALE == Scale::NONE)     return (T*) (((char*)addr)+shift);
        if constexpr(SCALE == Scale::GLOBAL)   return (T*) (((char*)addr)+shift+TENSOR_ALIGNMENT);
        if constexpr(SCALE == Scale::PER_COL)  return (T*) (((char*)addr)+shift+sizeof(float)*m);
        if constexpr(SCALE == Scale::PER_BLOC) return (T*) (((char*)addr)+shift+sizeof(float)*m*k/K1);
    }

private:
    const std::size_t _k;  // m contigue
    const std::size_t _m;
    const std::size_t _l;  // nb elements pour passer a la colonne (bloc) suivante si NATIF
    //const std::size_t m_f;  // nb elements pour passer a la colonne (bloc) suivante si NATIF
    // ??? doit-on gere les pilles de matices ici???

public:
    inline auto DIM1() const { return _k; }
    inline auto DIM2() const { return _m; }
    inline auto LD()   const {
        // static_assert(K1==0); // pour cela il faut savoir ou on est... en k
        return _l;
    }

    T* m_values;             // suivant le format ca peut-etre une copie!
    T* m_values_fin=nullptr; // le dernier bloc si K1>0 et k%K1 != 0
    float* m_scale=nullptr;

    inline Matrice(T* v, std::size_t k, std::size_t m, std::size_t l): _k(k), _m(m), _l(next_bloc(l,_k,_m)),
            m_values(data_adr(v,_k,_m)),
            m_scale(scale_adr(v))
    {
        static_assert(K0>0);
        static_assert(SCALE==Scale::NONE);
        if constexpr (K0>1) GGML_ASSERT(_k%K0==0);
        if constexpr (M0>1) GGML_ASSERT(_m%M0==0);
        if constexpr (K1>0) GGML_ASSERT(_k%K1==0); // pour l'instant
    }

    // un buffer avec le meme format que le tenseur
    inline Matrice(const void * data, struct ggml_tensor * t):
            _k(t->ne[0]), _m(t->ne[1]),
            _l(next_bloc((t->nb[1]/t->nb[0]),_k,_m)),
            m_values(data_adr(data,_k,_m)),
            m_scale(scale_adr(data))
    {
        static_assert(K0>0);
        static_assert(NATIF);
        GGML_ASSERT(t->ne[2]==1);
        GGML_ASSERT(t->ne[3]==1);
        if constexpr (K0>1) GGML_ASSERT(_k%K0==0);
    }

    inline Matrice(struct ggml_tensor * t):
            _k(t->ne[0]), _m(t->ne[1]),
            _l(next_bloc((t->nb[1]/t->nb[0]),_k,_m)),
            m_values(data_adr(t->data,_k,_m)),
            m_scale(scale_adr(t->data))
    {
        static_assert(K0>0);
        GGML_ASSERT(t->ne[2]==1);
        GGML_ASSERT(t->ne[3]==1);
    }

    inline Matrice(struct ggml_tensor * t, std::size_t i, std::size_t j): _k(t->ne[0]), _m(t->ne[1]), _l(t->nb[1]/t->nb[0]),
            m_values(data_adr(t->data, t->ne[1], t->ne[1], i*t->nb[2]+j*t->nb[3])),
            m_scale(scale_adr(t->data))
    {
        // pas (encore) utilisable pour les empilements de matrice non native!
        static_assert(NATIF);
        //if constexpr(!NATIF) {
        //    GGML_ASSERT(i==0);
        //    GGML_ASSERT(j==0);
        //}
    }

    inline void set_scale(float val, std::size_t j=0, std::size_t k=0) {
        if constexpr(SCALE==Scale::GLOBAL)  *m_scale = val;
        if constexpr(SCALE==Scale::PER_COL)  m_scale[j] = val;
        if constexpr(SCALE==Scale::PER_BLOC) m_scale[_m*(k/K1)+j] = val;
    }

    inline auto get_scale(std::size_t j=0, std::size_t k=0) const {
        if constexpr(SCALE==Scale::NONE) return (float)1;
        if constexpr(SCALE==Scale::GLOBAL) return *m_scale;
        if constexpr(SCALE==Scale::PER_COL) return m_scale+j;
        if constexpr(SCALE==Scale::PER_BLOC) return m_scale+j+_m*(k/K1);
    }
    template<int N>
    inline auto get_scale_v(std::size_t j, std::size_t k=0) const {
        if constexpr(SCALE==Scale::NONE)     return _mm512_set1_ps ((float)1);
        if constexpr(SCALE==Scale::GLOBAL)   return _mm512_set1_ps (*m_scale);
        if constexpr(SCALE==Scale::PER_COL)  return load<typename type_t<fp32_t, N>::t>(m_scale+j); // float[j,...]
        if constexpr(SCALE==Scale::PER_BLOC) return load<typename type_t<fp32_t, N>::t>(m_scale+j+_m*(k/K1)); // float[j,...]
    }

    // get_scale<vector>
    // calcule la position de l'element dans le tableau...
    template<bool BLOC>
    std::size_t inline indice(std::size_t k, std::size_t i) const {
        if constexpr(NATIF) {
            return i*_l+k;
        } else if constexpr(K1==0) {
            const auto j0 = i%M0; const auto j1 = i/M0;
            auto indice = j1*_l+j0*K0;
            if constexpr(!BLOC) {
                const auto i0 = k%K0; const auto i1 = k/K0;
                indice += i1*K0*M0+i0;
            }
            return indice;
        } else {
            // on a 1 K1...             return m_values+i1*_l+k1*K0*M0;
            const auto i1 = i/M0;
            const auto k2 = k/K1;
            auto indice = k2*_l+i1*K1*M0;
            if constexpr(!BLOC) {
                const auto k0 = k%K0;
                const auto k1 = (k/K0)%(K1/K0);  // (k/K0)%((_k%K1)/K0) pour le dernier bloc si pas complet?
                const auto i0 = i%M0;
                indice += k1*M0*K0+i0*K0+k0;
            } else {
                GGML_ASSERT(k%K1==0);
                GGML_ASSERT(i%M0==0);
                GGML_ASSERT((k/K0)%(K1/K0)==0);
            }
            //  A[K/K0*K1][M/M0][K1/K0][M0][K0]
            //      k2      i1    k1    i0  k0
            // ?? => le dernier bloc K == [K0][M0][K%(K0*K1)][M/M0] ...
            return indice;
        }
    }

    inline T& operator()(std::size_t k, std::size_t i) {
        return m_values[indice<false>(k,i)];
    }
    inline const T operator()(std::size_t k, std::size_t i) const {
        return m_values[indice<false>(k,i)];
    }
    inline T* addr(std::size_t k, std::size_t i) {
        GGML_ASSERT(k<_k);
        GGML_ASSERT(i<_m);
        return m_values + indice<true>(k,i);
    }
    inline const T* addr(std::size_t k, std::size_t i) const {
        GGML_ASSERT(k<_k);
        GGML_ASSERT(i<_m);
        return m_values + indice<true>(k,i);
    }

    inline float max() const {
        float res = 0;
        float val;
#pragma omp parallel for schedule(guided)
        for (std::size_t i=0; i<_m; i++) {
            for (std::size_t k=0; k<_k; k++) {
                conv(val, (*this)(k,i));
                val = val>0?val:-val;
                if (val>res) res = val;
            }
        }
        return res;
    }
    inline float max(std::size_t i) const {
        float res = 0;
        float val;
        for (std::size_t k=0; k<_k; k++) {
            conv(val, (*this)(k,i));
            val = val>0?val:-val;
            if (val>res) res = val;
        }
        return res;
    }
    template<std::size_t SIZE>
    inline float max(std::size_t k0, std::size_t i) const {
        float res = 0;
        float val;
        const auto end = std::min(_k, SIZE+k0);
        GGML_ASSERT(end == k0+SIZE);
        for (std::size_t k=k0; k<end; k++) {
            conv(val, (*this)(k,i));
            val = val>0?val:-val;
            if (val>res) res = val;
        }
        return res;
    }

};
