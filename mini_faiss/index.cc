#include "index.h"
#include <vector>
namespace miniFaiss {
void Index::Assign(idx_t n, const faiss_float *x, idx_t *labels,
                   idx_t k) const {
  std::vector<faiss_float> distances(n * k);
  Search(n, x, k, distances.data(), labels);
}

void Index::reconstructBatch(idx_t n, const idx_t *keys,
                             faiss_float *recons) const {
#ifdef USE_MKL
#pragma omp parallel for if (n > 1000)
#endif
  for (idx_t i = 0; i < n; ++i) {
    reconstruct(keys[i], &recons[i * d_]);
  }
}

} // namespace miniFaiss
