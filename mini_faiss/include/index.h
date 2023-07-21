#pragma once

#include "metricType.h"
#include <omp.h>
namespace miniFaiss {
struct IDSelector;
struct RangeSearchResult;
struct DistanceComputer;

struct SearchParameters {
  IDSelector *selector = nullptr;
  virtual ~SearchParameters() {}
};

class Index {
  using component_t = float;
  using distance_t = float;

public:
  /**
   * training on a representative set of vectors
   * @param n number of training vectors
   * @param x training vectors(size: n * d_)
   */
  virtual void Train(idx_t n, const faiss_float *x) {}

  /**
   * Add n vectors of dimension d_ to the index
   * @param x input matrix(size: n * d_)
   */
  virtual void Add(idx_t n, const faiss_float *x) = 0;

  /**
   * Same as Add, but stores xids instead of sequential ids.
   * @param xids if non-null, ids to store for the vectors (size n)
   */
  virtual void AddWithIds(idx_t n, const faiss_float *x, const idx_t *xids) = 0;

  /**
   * Query n vectors of dimension d_ to the index
   * return at most k vectors. If there are not enough
   * results for a query, the result array is padded
   * with -1s.
   * @param x input vectors to search, size n * d_
   * @param labels output labels of the NNs, size n * k
   * @param distances output pairwise distance, size n * k
   */
  virtual void Search(idx_t n, const faiss_float *x, idx_t k,
                      faiss_float *distances, idx_t *lables,
                      const SearchParameters *params = nullptr) const = 0;
  /**
   * Query n vectors of dimension d_ to the index
   * return all vectors with distance < radius. Note that
   * many indexes do not implement the RangeSearch (only the
   * K-NN search is mandatory)
   * @param x input vectors to search, size n * d_
   * @param radius search radius
   * @param result result table
   */
  virtual void RangeSearch(idx_t n, const faiss_float *x, faiss_float radius,
                           RangeSearchResult *result,
                           const SearchParameters *params = nullptr) const = 0;

  /**
   * Return the indexes of the k vectors closest to the query x.
   * This function is identical as search but only return labels of neighbors.
   * @param x input vectors to search, size n * d_
   * @param label output labels of the NNs, size n * k
   */
  virtual void Assign(idx_t n, const faiss_float *x, idx_t *labels,
                      idx_t k = 1) const;
  // Removes all elements from the database.
  virtual void Reset() = 0;

  /**
   * Removes IDs from the index. Not supported by 
   * all indexes. Returns the number of elements
   * removed.
   */
   virtual size_t RemoveIds(const IDSelector &sel) = 0;

protected:
   /**
    * Reconstruct a stored vector (or an approximation if lossy coding)
    * may not be adapted by all indexes
    * @param key id of the vector to reconstruct  
    * @param recons reconstructed vector (size d_)
    */
    virtual void reconstruct(idx_t key, faiss_float* recons) const = 0;

    virtual void reconstructBatch(idx_t n, const idx_t* key, faiss_float* recons) const;

    virtual void reconstructN(idx_t i0, idx_t ni, faiss_float* recons) const;
protected:
  int d_;        // vector dimentsion
  idx_t ntotal_; // total nb of indexed vectors
  bool is_trained_;
  bool verbose __attribute__((maybe_unused));
  MetricType metric_type_;
  faiss_float metric_arg_;
};

} // namespace miniFaiss