//------------------------------------------------------------------------------------
// gestion de Matrice (en attandant mieux)
// @ mettre au propre: K<>m M<>n
// => TODO gerer des bloc de taille [K1,M1,N1] ??? ...

enum class Scale {
    NONE,
    GLOBAL,
    PER_COL
};

namespace std {
constexpr std::string to_string(Scale s) {
    if (s==Scale::NONE) return "NONE";
    if (s==Scale::GLOBAL) return "GLOBAL";
    if (s==Scale::PER_COL) return "PER_COL";
    return "UNKOW";
}
}

// stockage @ gerer:
//  A[ 0][ 0][ 0][K][M]
//  A[ 1][ 1][ 0][K][M] ???
//  A[K0][ 1][ 0][K/K0][M]
//  A[K0][M0][ 0][K/K0][M/M0]
//  A[K0][M0][K1][M/M0][K/K0*K1]  ??? que faire si  K%(K0*K1)!=0
//     => le dernier bloc K == [K0][M0][K%(K0*K1)][M/M0] ...

//template<typename T, size_t K0=1, size_t M0=1, size_t K1=0, Scale SCALE=Scale::NONE>
template<typename T, size_t K0=1, size_t M0=1, Scale SCALE=Scale::NONE>
class Matrice {
public:
    // qq-aides:
    static inline std::size_t scale_size(std::size_t n) {
        if constexpr(SCALE == Scale::NONE)    return 0;
        if constexpr(SCALE == Scale::GLOBAL)  return TENSOR_ALIGNMENT;
        if constexpr(SCALE == Scale::PER_COL) return sizeof(float)*n;
    }

    static inline float* scale_adr(const void * data) {
        if constexpr(SCALE == Scale::NONE) {
            return nullptr;
        } else {
            return (float*)data;
        }
    }

    static inline T* data_adr(const void * addr, std::size_t n, std::size_t shift = 0) {
        if constexpr(SCALE == Scale::NONE)    return (T*) (((char*)addr)+shift);
        if constexpr(SCALE == Scale::GLOBAL)  return (T*) (((char*)addr)+shift+TENSOR_ALIGNMENT);
        if constexpr(SCALE == Scale::PER_COL) return (T*) (((char*)addr)+shift+sizeof(float)*n);
    }

public:
    inline auto DIM1() const { return m_m; }
    inline auto DIM2() const { return m_n; }
    inline auto LD()   const { return m_l; }

    const std::size_t m_m;  // m contigue
    const std::size_t m_n;
    const std::size_t m_l;  // nb elements pour passer a la colonne (bloc) suivante.
    T* m_values;  // suivant le format ca peut-etre une copie!
    float* m_scale=nullptr;

    inline Matrice(T* v, std::size_t m, std::size_t n, std::size_t l): m_m(m), m_n(n), m_l((K0*M0==1)?l:m*K0*M0),
            m_values(data_adr(v,n)),
            m_scale(scale_adr(v))
    {
        static_assert(SCALE==Scale::NONE);
        static_assert(K0>0);
        static_assert(M0>0);
    }

    // un buffer avec le meme format que le tenseur
    inline Matrice(const void * data, struct ggml_tensor * t):
            m_m(t->ne[0]), m_n(t->ne[1]),
            m_l((t->nb[1]/t->nb[0])),
            m_values(data_adr(data,t->ne[1])),
            m_scale(scale_adr(data))
    {
        static_assert(K0==1);
        static_assert(M0==1);
        GGML_ASSERT(t->ne[2]==1);
        GGML_ASSERT(t->ne[3]==1);
    }

    inline Matrice(struct ggml_tensor * t):
            m_m(t->ne[0]), m_n(t->ne[1]),
            m_l((K0*M0==1)?(t->nb[1]/t->nb[0]):t->ne[0]*M0),
            m_values(data_adr(t->data,t->ne[1])),
            m_scale(scale_adr(t->data))
    {
        static_assert(K0>0);
        static_assert(M0>0);
        GGML_ASSERT(t->ne[2]==1);
        GGML_ASSERT(t->ne[3]==1);
        //GGML_ASSERT((sizeof(T)*m_m*m_n)<=ggml_nbytes(t));
        // std::cout <<" Matrice: "<< ":" <<m_m<<"/"<<m_n<<"/"<<m_l <<" | "<< t << std::endl;
    }

    inline Matrice(struct ggml_tensor * t, std::size_t i, std::size_t j): m_m(t->ne[0]), m_n(t->ne[1]), m_l(t->nb[1]/t->nb[0]),
            m_values(data_adr(t->data, t->ne[1], i*t->nb[2]+j*t->nb[3])),
            m_scale(scale_adr(t->data))
    {
        // pas (encore) utilisable pour les empilements de matrice non native!
        static_assert(M0==1);
    }

    inline void set_scale(float val, size_t j=0) {
        if constexpr(SCALE==Scale::GLOBAL) *m_scale = val;
        if constexpr(SCALE==Scale::PER_COL) m_scale[j] = val;
    }

    inline auto get_scale(size_t j=0) const {
        if constexpr(SCALE==Scale::GLOBAL) return *m_scale;
        if constexpr(SCALE==Scale::PER_COL) return m_scale+j; // float[j,...]
        if constexpr(SCALE==Scale::NONE) return (float)1;
    }
    template<int N>
    inline auto get_scale_v(size_t j) const {
        static_assert(SCALE==Scale::PER_COL); // voir pour les autres....
        if constexpr(SCALE==Scale::GLOBAL)  return _mm512_set1_ps (*m_scale);
        if constexpr(SCALE==Scale::PER_COL) return load<typename type_t<fp32_t, N>::t>(m_scale+j); // float[j,...]
        if constexpr(SCALE==Scale::NONE)    return _mm512_set1_ps ((float)1);
    }

    // get_scale<vector>

    inline T& operator()(size_t i, size_t j) {
        const auto i0 = i%K0; const auto i1 = i/K0;
        const auto j0 = j%M0; const auto j1 = j/M0;
        return m_values[j1*m_l+j0*K0+i1*K0*M0+i0];
    }
    inline const T operator()(size_t i, size_t j) const {
        const auto i0 = i%K0; const auto i1 = i/K0;
        const auto j0 = j%M0; const auto j1 = j/M0;
        return m_values[j1*m_l+j0*K0+i1*K0*M0+i0];
    }

    inline T* addr(size_t i, size_t j) {
        if constexpr(K0>1 || M0>1) {
            // pas une bonne idée de taper en dehors d'un bloc complet,
            //GGML_ASSERT(i<m_m);
            //GGML_ASSERT(j<m_n);
            //GGML_ASSERT(j%M==0);
            //GGML_ASSERT(i%K==0);
            //GGML_ASSERT(j%M==0);
            const auto i1 = i/K0;
            const auto j1 = j/M0;
            return m_values+j1*m_l+i1*K0*M0;
        } else {
            return m_values+j*m_l+i; // [j1*m_l+j0*K+i1*K*M+i0]
        }
    }
    inline const T* addr(size_t i, size_t j) const {
        if constexpr(K0>1 || M0>1) {
            // pas une bonne idée de taper en dehors d'un bloc complet,
            //GGML_ASSERT(i%K==0);
            //GGML_ASSERT(j%M==0);
            const auto i1 = i/K0;
            const auto j1 = j/M0;
            return m_values+j1*m_l+i1*K0*M0;
        } else {
            return m_values+j*m_l+i; // [j1*m_l+j0*K+i1*K*M+i0]
        }
    }

    inline float max() const {
        float res = 0;
        float val;
#pragma omp parallel for schedule(guided)
        for (size_t j=0; j<m_n; j++) {
            for (size_t i=0; i<m_m; i++) {
                conv(val, (*this)(i,j));
                val = val>0?val:-val;
                if (val>res) res = val;
            }
        }
        return res;
    }
    inline float max(size_t j) const {
        float res = 0;
        float val;
        for (size_t i=0; i<m_m; i++) {
            conv(val, (*this)(i,j));
            val = val>0?val:-val;
            if (val>res) res = val;
        }
        return res;
    }

};
