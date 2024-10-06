//------------------------------------------------------------------------------------
// gestion de Matrice (en attandant mieux)
// @ mettre au propre: K<>m M<>n
// => TODO gerer des bloc de taille [K1,M1,N1] ??? ...

enum class Scale {
    NONE,
    BLOC,  // fct de KB/MB.
};

namespace std {
constexpr std::string to_string(Scale s) {
    if (s==Scale::NONE) return "NONE";
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
    // format natif => valeur[m][l] avec l>=k
    static constexpr bool NATIF = (M0==0)&&(K1==0); // && SCALE==Scale::NONE ???
    Matrice()=delete;
    Matrice(Matrice&)=delete;
    Matrice(const Matrice&)=delete;
    Matrice(Matrice&&)=delete;
private:
    static void control() {
        // qq elements sont a respecter:
        static_assert(K0>0);
        static_assert(K1==0 || K1%K0==0);
        static_assert((KB==0 && MB==0) || SCALE==Scale::BLOC); // Scale::NONE => KB=MB=0
        static_assert(M0==0 || MB==0 || MB%M0==0 || MB==1);  // MB==1 => load<M0>, MB%M0==0 => brodcast<M0>
        static_assert(KB==0 || KB%K0==0);
        static_assert(K1==0 || KB==0 || K1%KB==0 || KB%K1==0);
        //static_assert(KB!=0 || MB!=0 || SCALE!=Scale::BLOC);
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
        if constexpr(SCALE == Scale::BLOC) {
            if constexpr(MB==0 && KB==0) return TENSOR_ALIGNMENT; // facteur GLOBAL => voir si on le garde ?
            if constexpr(KB==0) return sizeof(float)*(((m-1)/MB)+1);
            if constexpr(MB==0) return sizeof(float)*(((k-1)/KB)+1);
            if constexpr(MB!=0 && KB!=0) return sizeof(float)*(((k-1)/KB)+1)*(((m-1)/MB)+1);
        }
    }
    static inline std::size_t scale_size(const struct ggml_tensor * t) {
        return scale_size(t->ne[0], t->ne[1]);
    }

    // inline Matrice(struct ggml_tensor * t):
    static inline std::size_t size(const struct ggml_tensor * t) {
        // TODO: vrai natif: return ggml_nbytes(t);
        if constexpr (NATIF) {
            // il sera convertit / paqué ...etc
            GGML_ASSERT(t->ne[2]==1);
            GGML_ASSERT(t->ne[3]==1);
            // pour rester coherant avec le calcul de l ...
            auto size = sizeof(T)*(t->nb[1]/t->nb[0])*t->ne[1]; //*tensor->ne[2]*tensor->ne[3];
            return size+scale_size(t);
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

    static std::size_t l_scale(std::size_t k, std::size_t m) {
        if constexpr (MB==0 || KB==0) return 1;      // 1 valeur ou 1 vecteur... => pas utilisé!
        if constexpr (MB==1) return (((m-1)/MB)+1);  // les m doivent se suivrent...
        if constexpr (KB>0)  return (((k-1)/KB)+1);  // les k doivent se suivrent
    }

    inline Matrice(T* v, std::size_t k, std::size_t m, std::size_t l):
            _k(k), _m(m),
            _l(next_bloc(l,k,m)),
            _sl(l_scale(k,m)), // KB>0 => vrai bloc. scale[m/MB][k/KB]
            m_values(data_adr(v,k,m)),
            m_scale(scale_adr(v))
    {
        control();
        // KB>0 => MB >0
        static_assert(KB==0 || MB>0);
    }
public:

    // pour les set_tensor (un buffer avec le meme format que le tenseur)
    inline Matrice(const void * data, struct ggml_tensor * t): Matrice((T*)data,t->ne[0],t->ne[1],t->nb[1]/t->nb[0])
            //_k(t->ne[0]),_m(t->ne[1]),_l(next_bloc((t->nb[1]/t->nb[0]),_k,_m)),m_values(data_adr(data,_k,_m)),m_scale(scale_adr(data))
    {
        control();
        static_assert(NATIF);
        GGML_ASSERT(t->ne[2]==1);
        GGML_ASSERT(t->ne[3]==1);
        if constexpr (K0>1) GGML_ASSERT(_k%K0==0);
    }

    // les autres gemm..
    inline Matrice(struct ggml_tensor * t): Matrice((T*)t->data,t->ne[0],t->ne[1],t->nb[1]/t->nb[0])
            //_k(t->ne[0]),_m(t->ne[1]),_l(next_bloc((t->nb[1]/t->nb[0]),_k,_m)),m_values(data_adr(t->data,_k,_m)),m_scale(scale_adr(t->data))
    {
        control();
        GGML_ASSERT(t->ne[2]==1);
        GGML_ASSERT(t->ne[3]==1);
        // std::cout<< "Matrice<"<<t->name<<">("<<_k<<","<<_m<<")"<<std::endl;
    }

    // sgemm seulement pour l'instant!
    inline Matrice(struct ggml_tensor * t, std::size_t i, std::size_t j):
            _k(t->ne[0]), _m(t->ne[1]),
            _l(t->nb[1]/t->nb[0]),
            m_values(data_adr(t->data, _k, _m, i*t->nb[2]+j*t->nb[3])),
            m_scale(scale_adr(t->data)) // nullptr !!
    {
        control();
        // pas (encore) utilisable pour les empilements de matrice non native!
        static_assert(NATIF);
        if constexpr(SCALE!=Scale::NONE) { // ?? p'etre possible en GLOBAL...
            GGML_ASSERT(i==0);
            GGML_ASSERT(j==0);
        }
    }

private:
    inline std::size_t indice_scale(std::size_t k, std::size_t i) const {
        static_assert(SCALE==Scale::BLOC);
        if constexpr(SCALE==Scale::BLOC) {
            if constexpr(MB==0 && KB==0) return 0;       // facteur GLOBAL => voir si on le garde ?
            if constexpr(KB==0)          return i/MB;    // toutes les MB collones
            if constexpr(MB==0)          return k/KB;    // toutes les KB lignes
            if constexpr(MB==1 && KB!=0) return _sl*(k/KB) + i;      // ici les m doivent se suivre ... _sl = ((k-1)/KB)+1 ???
            if constexpr(MB>1 && KB!=0)  return (k/KB) + _sl*(i/MB); // _sl = ((k-1)/KB)+1 ???
        }
    }

public:
    inline void set_scale(float val, std::size_t k, std::size_t i) {
        if constexpr(SCALE==Scale::BLOC) m_scale[indice_scale(k,i)] = val;
    }

    /*
    inline auto get_scale(std::size_t j, std::size_t k) const {
        if constexpr(SCALE==Scale::NONE) {
            return (float)1;
        } else {
            return m_scale[indice_scale(k,j)];
        }
    }
    */
    // pour sgemm ... en attandant de faire ca avec addr_scale
    inline auto get_scale(std::size_t I0) const {
        if constexpr(SCALE==Scale::NONE) {
            return (float)1;
        } else if constexpr(SCALE==Scale::BLOC && KB==0 && MB==0) {
            return *m_scale;
        } else if constexpr(SCALE==Scale::BLOC && KB==0 && MB==1) {
            return m_scale+indice_scale(0,I0);
        } else {
            static_assert(false, "Si besion il faut reformater avec addr_scale!");
        }
    }

    inline auto addr_scale(std::size_t k0, std::size_t i0) {
        static_assert(SCALE==Scale::BLOC);
        if constexpr(SCALE==Scale::BLOC) {
            return m_scale+indice_scale(k0,i0);
        }
    }

    inline const float* addr_scale(std::size_t k0, std::size_t i0) const {
        static_assert(SCALE==Scale::BLOC);
        if constexpr(SCALE==Scale::BLOC) {
            return m_scale+indice_scale(k0,i0);
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
                //std::cout << "k/i = "<< k << "/"<<i<<":"<<K1<<"/"<<M0<<std::endl;
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
        //std::cout << " addr("<<k<<","<<i<<") : "<<_k<<"/"<<_m<<std::endl;
        GGML_ASSERT(k<_k);
        GGML_ASSERT(i<_m);
        return m_values + indice<true>(k,i);
    }

    inline float max() const {
        static_assert(SCALE==Scale::NONE); // sinon c'est "debile"
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
    template<int K, int M>
    inline float max(std::size_t k0, std::size_t i0) const {
        static_assert(SCALE==Scale::NONE); // sinon c'est "debile"
        static_assert(K!=0 || M!=0); // max global
        if constexpr(K==0) { // sur tout la colonne
            GGML_ASSERT(k0==0);
            GGML_ASSERT(i0%M==0); // aligné sur la grille
            float res = 0;
            float val;
            const auto m_end = std::min(_m, M+i0);
            for (std::size_t i=i0; i<m_end; i++)
            for (std::size_t k=0; k<_k; k++) {
                conv(val, (*this)(k,i0));
                val = val>0?val:-val;
                if (val>res) res = val;
            }
            return res;
        }
        if constexpr(M==0) { // sur tout la ligne (des fois que... mais peu de chance)
            GGML_ASSERT(i0==0);
            GGML_ASSERT(k0%K==0); // aligné sur la grille
            float res = 0;
            float val;
            const auto k_end = std::min(_k, K+k0);
            for (std::size_t k=k0; k<k_end; k++)
            for (std::size_t i=0; i<_m; i++) {
                conv(val, (*this)(k0,i));
                val = val>0?val:-val;
                if (val>res) res = val;
            }
            return res;
        }
        if constexpr(K>0 && M>0) { // de "vrai" blocs
            GGML_ASSERT(k0%K==0); // aligné sur la grille
            GGML_ASSERT(i0%M==0); // aligné sur la grille
            float res = 0;
            float val;
            const auto k_end = std::min(_k, K+k0);
            const auto m_end = std::min(_m, M+i0);
            for (std::size_t i=i0; i<m_end; i++)
            for (std::size_t k=k0; k<k_end; k++) {
                conv(val, (*this)(k,i));
                val = val>0?val:-val;
                if (val>res) res = val;
            }
            return res;
        }
    }

};
