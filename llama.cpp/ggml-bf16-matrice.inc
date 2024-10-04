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
    // PER_BLOC,  // la taille des bloc de K1.
    BLOC,  // fct de KB/MB.
};

namespace std {
constexpr std::string to_string(Scale s) {
    if (s==Scale::NONE) return "NONE";
    if (s==Scale::GLOBAL) return "GLOBAL";
    if (s==Scale::PER_COL) return "PER_COL";
    if (s==Scale::BLOC) return "BLOC";
    return "UNKNOW";
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

// scale par bloc:
//  KB=MB=0 => global (meme si ca marche pas trop...)
//  KB=0    => 1 facteur pour des blok A[MB][:]   => par colonne
//  MB=0    => 1 facteur pour des blok A[:][KB]   => pas a faire
//          => 1 facteur pour des blok A[MB][KB]  => par bloc @ tester mais surement le mieux
// TODO: voir comment gere les cas d'emplimement de matrice (?)
template<typename T,
std::size_t K0=1, std::size_t M0=0, std::size_t K1=0,
Scale SCALE=Scale::NONE, std::size_t MB=0, std::size_t KB=0
>
class Matrice {
    static constexpr bool NATIF = (M0==0)&&(K1==0); // && SCALE==Scale::NONE ???
private:
    enum class PRIVATE { NONE };
    static void control() {
        // qq elements sont a respecter:
        static_assert(K0>0);
        static_assert(K1==0 || K1%K0==0);
        static_assert((KB==0 && MB==0) || SCALE==Scale::BLOC);
        static_assert(M0==0 || MB==0 || MB%M0==0 || MB==1);  // MB==1 => load<M0>, MB%M0==0 => brodcast<M0>
        static_assert(KB==0 || KB%K0==0);
        static_assert(K1==0 || KB==0 || K1%KB==0 || KB%K1==0);
        static_assert(KB!=0 || MB!=0 || SCALE!=Scale::BLOC);
        //static_assert(KB!=0 || MB!=0); // voir a interdire le scale GLOBAL
    }
public:
    // qq-aides:
    static inline std::size_t next_bloc(std::size_t l, std::size_t k, std::size_t m) {
        return NATIF?l:(K1==0?k*M0:m*K1);
    }
    // la taille necessaire pour le bloc de scale en octet!
    static inline std::size_t scale_size(std::size_t k, std::size_t m) {
        if constexpr(SCALE == Scale::NONE)    return 0;
        if constexpr(SCALE == Scale::GLOBAL)  return TENSOR_ALIGNMENT;
        if constexpr(SCALE == Scale::PER_COL) return sizeof(float)*m;
        if constexpr(SCALE == Scale::BLOC) {
            if constexpr(MB==0 && KB==0) TENSOR_ALIGNMENT; // facteur GLOBAL => voir si on le garde ?
            if constexpr(KB==0) return sizeof(float)*(((m-1)/MB)+1);
            if constexpr(MB==0) return sizeof(float)*(((k-1)/KB)+1);
            if constexpr(MB!=0 && KB!=0) sizeof(float)*(((k-1)/KB)+1)*(((m-1)/MB)+1);
        }
    }
    static inline std::size_t scale_size(const struct ggml_tensor * t) {
        return scale_size(t->ne[0], t->ne[1]);
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
        return (T*) (((char*)addr)+shift+scale_size(k,m));
    }

public:
    inline auto DIM1() const { return _k; }
    inline auto DIM2() const { return _m; }
    inline auto LD()   const {
        // static_assert(K1==0); // pour cela il faut savoir ou on est... en k
        return _l;
    }

private:
    const std::size_t _k;  // dimension contigue
    const std::size_t _m;
    const std::size_t _l;  // nb elements pour passer a la colonne (bloc) suivante si NATIF
    const std::size_t _sl=1; // le nombre de scale factor suivant K (si "vrai" bloc) = ((k-1)/KB)+1
    // ??? doit-on gere les pilles de matices ici???

    T* m_values;             // suivant le format ca peut-etre une copie!
    T* m_values_fin=nullptr; // le dernier bloc si K1>0 et k%K1 != 0
    float* m_scale=nullptr;

    inline Matrice(PRIVATE, void* v, std::size_t k, std::size_t m, std::size_t l):
            _k(k), _m(m),
            _l(next_bloc(l,_k,_m)),
            _sl(KB>0?(((_k-1)/KB)+1):1), // dans tous les cas contigue suivant _m (load comme vecteur)
            m_values(data_adr(v,_k,_m)),
            m_scale(scale_adr(v))
    {
        control();
    }
public:
    inline Matrice(T* v, std::size_t k, std::size_t m, std::size_t l): Matrice(PRIVATE::NONE, v, k, m, l)/*,
            _k(k), _m(m),
            _l(next_bloc(l,_k,_m)),
            m_values(data_adr(v,_k,_m)),
            m_scale(scale_adr(v))
            */
    {
        control();
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
        control();
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
        control();
        GGML_ASSERT(t->ne[2]==1);
        GGML_ASSERT(t->ne[3]==1);
    }

    inline Matrice(struct ggml_tensor * t, std::size_t i, std::size_t j):
            _k(t->ne[0]), _m(t->ne[1]),
            _l(t->nb[1]/t->nb[0]),
            m_values(data_adr(t->data, t->ne[1], t->ne[1], i*t->nb[2]+j*t->nb[3])),
            m_scale(scale_adr(t->data))
    {
        control();
        // pas (encore) utilisable pour les empilements de matrice non native!
        static_assert(NATIF);
        //if constexpr(!NATIF) {
        //    GGML_ASSERT(i==0);
        //    GGML_ASSERT(j==0);
        //}
    }

    inline void set_scale(float val, std::size_t j=0, std::size_t k=0) {
        if constexpr(SCALE!=Scale::NONE) m_scale[indice_scale(k,j)] = val;
    }

    inline auto get_scale(std::size_t j=0, std::size_t k=0) const {
        if constexpr(SCALE==Scale::NONE) {
            return (float)1;
        } else {
            return m_scale[indice_scale(k,j)];
        }
        //if constexpr(SCALE==Scale::NONE) return (float)1;
        //if constexpr(SCALE==Scale::GLOBAL) return *m_scale;
        //if constexpr(SCALE==Scale::PER_COL) return m_scale+j;
        //if constexpr(SCALE==Scale::PER_BLOC) return m_scale+j+_m*(k/K1);
    }
    template<int N>
    inline auto get_scale_v(std::size_t j, std::size_t k=0) const {
        if constexpr(SCALE==Scale::NONE)     return _mm512_set1_ps ((float)1);
        if constexpr(SCALE==Scale::GLOBAL)   return _mm512_set1_ps (*m_scale);
        if constexpr(SCALE==Scale::PER_COL)  return load<typename type_t<fp32_t, N>::t>(m_scale+j); // float[j,...]
        //if constexpr(SCALE==Scale::PER_BLOC) return load<typename type_t<fp32_t, N>::t>(m_scale+j+_m*(k/K1)); // float[j,...]
        if constexpr(SCALE==Scale::BLOC) {
            if constexpr(MB==0 && KB==0)  {
                return _mm512_set1_ps (*m_scale); // GLOBAL => brodcast
            }
            else if constexpr(MB==1 && KB==0)  {
                return load<typename type_t<fp32_t, N>::t>(m_scale+j); // PAR COLLONE => les N qui se suivent
            }
            else if constexpr(MB==1 && KB==K1) {
                return load<typename type_t<fp32_t, N>::t>(m_scale+j); // PAR BLOK(K1) de COLLONE => les N qui se suivent
            }
            else if constexpr(MB%M0==0 && KB>0) {
                // les cas BLOC[m/MB][k/KN] => 1 valeur a broadcaster
                _mm512_set1_ps (m_scale[indice_scale(k,j)]);
            } else {
                static_assert(false, "cas pas encore supporté");
            }

        }
    }

    inline std::size_t indice_scale(std::size_t k, std::size_t i) const {
        static_assert(SCALE!=Scale::NONE);
        if constexpr(SCALE==Scale::GLOBAL)   return 0;
        if constexpr(SCALE==Scale::PER_COL)  return i;  // == KB=0, MB=1
        if constexpr(SCALE==Scale::BLOC) {
            //return load<typename type_t<fp32_t, N>::t>(m_scale+j+_m*(k/K1)); // float[j,...]
            if constexpr(MB==0 && KB==0) 0; // facteur GLOBAL => voir si on le garde ?
            if constexpr(KB==0) return i%MB;   // toute les MB collone
            if constexpr(MB==0) return k%KB;
            if constexpr(MB!=0 && KB!=0) (k%KB) + _sl*(i%MB); // _sl = ((k-1)/KB)+1 ???
        }
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
