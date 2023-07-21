#include "metricType.h"
#include <omp.h>
namespace miniFaiss{
struct IDSelector;
struct RangeSearchResult;
struct DistanceComputer;

struct SearchParameters{
    IDSelector* selector = nullptr;
    virtual ~SearchParameters() {}
};

class Index{
    using component_t = float;
    using distance_t = float;
 public:
    /**
     * training on a representative set of vectors
     * @param n number of training vectors
     * @param x training vectors(size: n * d_)
     */
    virtual void Train(idx_t n, const faiss_float* x);

    /**
     * Add n vectors of dimension d_ to the index
     * @param x input matrix(size: n * d_)
     */
    virtual void Add(idx_t n, const faiss_float* x) = 0;

     /**
      * Same as Add, but stores xids instead of sequential ids.
      * @param xids if non-null, ids to store for the vectors (size n)
      */
    virtual void AddWithIds(idx_t n, const faiss_float* x, const idx_t *xids);
      
    /**
     * Query n vectors of dimension d_ to the index
     */
    virtual void Search(idx_t n, const faiss_float* x, idx_t k, faiss_float* distances, idx_t *lables, const SearchParameters* params = nullptr) const = 0;
 protected:
    int d_; // vector dimentsion
    idx_t ntotal_; // total nb of indexed vectors
    bool is_trained_;
    bool verbose __attribute__((maybe_unused));
    MetricType metric_type_;
    faiss_float metric_arg_;
};

}