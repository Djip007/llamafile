//#define DO_TIMING
#define DO_LOGS

//------------------------------------------------------------------------------------
#ifdef DO_TIMING
#include <ctime>
struct mesure {
    struct timespec ts_0;
    void start() {
        clock_gettime(CLOCK_REALTIME, &ts_0);
    }
    double end() {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        auto _s  = ts.tv_sec - ts_0.tv_sec;
        auto _ns = ts.tv_nsec - ts_0.tv_nsec;
        return ((double) _s) + ((double) _ns)/1.e9;
    }
};
#endif // DO_TIMING

//------------------------------------------------------------------------------------
// Logs simple sur un flux c++
// log des tenseurs sources
struct log_srcs {
    log_srcs(const ggml_tensor * t): _t(t){}
    const ggml_tensor * _t;
};

#ifdef DO_LOGS
#include <iostream>
static inline std::ostream& operator<<(std::ostream& os, enum ggml_type type) {
    return os << ggml_type_name(type);
}

static inline std::ostream& operator<<(std::ostream& os, enum ggml_op type) {
    return os << ggml_op_name(type);
}

// log du tenseur destination
static inline std::ostream& operator<<(std::ostream& os, const struct ggml_tensor * t) {
    // TODO voir a afficher si c'est un tenseur/OP
    os <<"["<<t->ne[0];
    for (int i=1; i<GGML_MAX_DIMS ; i++) {
        os <<":"<<t->ne[i];
    }
    os <<"/"<<t->nb[0];
    for (int i=1; i<GGML_MAX_DIMS ; i++) {
        os <<":"<<t->nb[i];
    }
    return os <<"]@"<< t->type;
}

static inline std::ostream& operator<<(std::ostream& os, const struct log_srcs t0) {
    auto t = t0._t;
    if (t->src[0]) os<<"s0("<<t->src[0]->name<< (t->src[0]->extra==nullptr?"-":"+") <<")"<<t->src[0];
    for(int i=1; i<GGML_MAX_SRC; i++) {
        if (t->src[i]) os<<", s"<<i<<"("<<t->src[i]->name<<")"<<t->src[i];
    }
    return os;
}

#else
// desactivation des logs:
static inline std::ostream& operator<<(std::ostream& os, enum ggml_type type)          { return os; }
static inline std::ostream& operator<<(std::ostream& os, enum ggml_op type)            { return os; }
static inline std::ostream& operator<<(std::ostream& os, const struct ggml_tensor * t) { return os; }
static inline std::ostream& operator<<(std::ostream& os, const struct log_srcs t0)     { return os; }
#endif
