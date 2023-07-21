#pragma once
#include <cstdint>
#include <cstdio>
namespace miniFaiss {
enum MetricType {
  METRIC_INNER_PRODUCT = 0, ///< maximum inner product search
  METRIC_L2 = 1,            ///< squared L2 search
  METRIC_L1,                ///< L1 (aka cityblock)
  METRIC_Linf,              ///< infinity distance
  METRIC_Lp,                ///< L_p distance, p is given by a faiss::Index
                            /// metric_arg

  /// some additional metrics defined in scipy.spatial.distance
  METRIC_Canberra = 20,
  METRIC_BrayCurtis,
  METRIC_JensenShannon,
  METRIC_Jaccard, ///< defined as: sum_i(min(a_i, b_i)) / sum_i(max(a_i, b_i))
                  ///< where a_i, b_i > 0
};

using idx_t = int64_t;


/* 
 * this function is used to distinguish between min and max indexes since
 * we need to support similarity and dis-similarity metrics in a flexible way.
 */
constexpr bool is_similarity_metric(MetricType metric_type) {
    return ((metric_type == METRIC_INNER_PRODUCT) ||
            (metric_type == METRIC_Jaccard));
}

#ifdef USE_DOUBLE
  using faiss_float = double;
#else
  using faiss_float = float;
#endif
}