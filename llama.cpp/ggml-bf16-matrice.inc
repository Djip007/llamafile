//------------------------------------------------------------------------------------
// Tools
// - se deplacer dans un buffer en nombre de byte
static inline void* add_byte(void* addr, std::size_t nb) {
    return (void*) (((char*)addr)+nb);
}

//------------------------------------------------------------------------------------
// gestion de Matrice (en attandant mieux)
// @ mettre au propre: K<>m M<>n
// => TODO gerer des bloc de taille [K1,M1] ...
template<typename T, size_t K=1, size_t M=1>
class Matrice {
public:
    inline auto DIM1() const { return m_m; }
    inline auto DIM2() const { return m_n; }
    inline auto LD()   const { return m_l; }

    const std::size_t m_m;  // m contigue
    const std::size_t m_n;
    const std::size_t m_l;  // nb elements pour passer a la colonne (bloc) suivante.
    T* m_values;  // suivant le format ca peut-etre une copie!

    inline Matrice(T* v, std::size_t m, std::size_t n, std::size_t l): m_m(m), m_n(n), m_l((K*M==1)?l:m*K*M), m_values(v) {
        static_assert(K>0);
        static_assert(M>0);
    }

    // un buffer avec le meme format que le tenseur
    inline Matrice(const void* data, struct ggml_tensor * t):
            m_m(t->ne[0]), m_n(t->ne[1]),
            m_l((t->nb[1]/t->nb[0])),
            m_values((T*)(data))
    {
        static_assert(K==1);
        static_assert(M==1);
        GGML_ASSERT(t->ne[2]==1);
        GGML_ASSERT(t->ne[3]==1);
    }

    //inline Matrice(struct ggml_tensor * t): m_m(t->ne[0]), m_n(t->ne[1]), m_l((t->nb[1]/t->nb[0])*K*M), m_values((T*)(t->data)) {
    inline Matrice(struct ggml_tensor * t):
            m_m(t->ne[0]), m_n(t->ne[1]),
            m_l((K*M==1)?(t->nb[1]/t->nb[0]):t->ne[0]*M),
            m_values((T*)(t->data))
    {
        static_assert(K>0);
        static_assert(M>0);
        GGML_ASSERT(t->ne[2]==1);
        GGML_ASSERT(t->ne[3]==1);
        //GGML_ASSERT((sizeof(T)*m_m*m_n)<=ggml_nbytes(t));
        // std::cout <<" Matrice: "<< ":" <<m_m<<"/"<<m_n<<"/"<<m_l <<" | "<< t << std::endl;
    }

    inline Matrice(struct ggml_tensor * t, std::size_t i, std::size_t j): m_m(t->ne[0]), m_n(t->ne[1]), m_l(t->nb[1]/t->nb[0]) { //, m_values((T*)add_byte(t->data,)) {
        // pas utilisable pour les empilements de matrice non native!
        static_assert(K==1);
        static_assert(M==1);
        m_values = (T*) add_byte(t->data, i*t->nb[2]+j*t->nb[3]);
    }

    inline T& operator()(size_t i, size_t j) {
        const auto i0 = i%K; const auto i1 = i/K;
        const auto j0 = j%M; const auto j1 = j/M;
        return m_values[j1*m_l+j0*K+i1*K*M+i0];
        // return m_values[j*m_l+i];
    }
    inline const T operator()(size_t i, size_t j) const {
        const auto i0 = i%K; const auto i1 = i/K;
        const auto j0 = j%M; const auto j1 = j/M;
        return m_values[j1*m_l+j0*K+i1*K*M+i0];
        // return m_values[j*m_l+i];
    }

    inline T* addr(size_t i, size_t j) {
        if constexpr(K>1 || M>1) {
            // pas une bonne idée de taper en dehors d'un bloc complet,
            //GGML_ASSERT(i<m_m);
            //GGML_ASSERT(j<m_n);
            //GGML_ASSERT(j%M==0);
            //GGML_ASSERT(i%K==0);
            //GGML_ASSERT(j%M==0);
            const auto i1 = i/K;
            const auto j1 = j/M;
            return m_values+j1*m_l+i1*K*M;
        } else {
            return m_values+j*m_l+i; // [j1*m_l+j0*K+i1*K*M+i0]
        }
    }
    inline const T* addr(size_t i, size_t j) const {
        if constexpr(K>1 || M>1) {
            // pas une bonne idée de taper en dehors d'un bloc complet,
            //GGML_ASSERT(i%K==0);
            //GGML_ASSERT(j%M==0);
            const auto i1 = i/K;
            const auto j1 = j/M;
            return m_values+j1*m_l+i1*K*M;
        } else {
            return m_values+j*m_l+i; // [j1*m_l+j0*K+i1*K*M+i0]
        }
    }
};
