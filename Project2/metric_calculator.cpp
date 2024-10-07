#include "metric_calculator.h"
#include "utils.h"
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#define CHECK_VECTOR_LENGTH(a, b)                                              \
  do {                                                                         \
    if (a.size() != b.size()) {                                                \
      LOG_ERROR("vector size not matching: " << a.size() << " " << b.size());  \
      return std::numeric_limits<float>::infinity();                           \
    }                                                                          \
  } while (0)

const std::string kMetricKey = "metric";
const std::unordered_map<std::string,
                         std::function<float(const std::vector<float> &,
                                             const std::vector<float> &)>>
    kMetricCalculators = {{"SSD", sumOfSquaredDifference},
                          {"HistIntersection", histogramIntersection},
                          {"cosine", cosine},
                          {"crossEntropy", crossEntropy},
                          {"LN", LInfinityDistance}};

// util function
std::vector<float> normalize(const std::vector<float> &v) {
  float norm = std::sqrt(
      std::accumulate(v.begin(), v.end(), 0.0f, [](float ret, const auto &ele) {
        return ret + ele * ele;
      }));
  std::vector<float> normalizedVec(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    normalizedVec[i] = v[i] / norm;
  }
  return normalizedVec;
};

float sumOfSquaredDifference(const std::vector<float> &v1,
                             const std::vector<float> &v2) {
  CHECK_VECTOR_LENGTH(v1, v2);
  float result = 0.0f;
  for (size_t i = 0; i < v1.size(); ++i) {
    result += (v2[i] - v1[i]) * (v2[i] - v1[i]);
  }
  return result;
}

float histogramIntersection(const std::vector<float> &v1,
                            const std::vector<float> &v2) {
  // intersection: sum(min(v1, v2))
  CHECK_VECTOR_LENGTH(v1, v2);
  float result = 0;
  for (size_t i = 0; i < v1.size(); ++i) {
    result += std::min(v1[i], v2[i]);
  }
  return -result;
}

float cosine(const std::vector<float> &v1, const std::vector<float> &v2) {
  ASSERT_EQ(v1.size(), v2.size());
  std::vector<float> normVec1 = normalize(v1);
  std::vector<float> normVec2 = normalize(v2);
  float dotProduct = 0.0;
  for (size_t i = 0; i < normVec1.size(); ++i) {
    dotProduct += normVec1[i] * normVec2[i];
  }
  return 1 - dotProduct;
}

float crossEntropy(const std::vector<float> &v1, const std::vector<float> &v2) {
  const float kEpsilon = 1e-12;
  ASSERT_EQ(v1.size(), v2.size());
  std::vector<float> normVec1 = normalize(v1);
  std::vector<float> normVec2 = normalize(v2);
  float cross_entropy = 0.0f;
  for (size_t i = 0; i < normVec1.size(); ++i) {
    cross_entropy +=
        normVec1[i] * std::log(std::clamp(normVec2[i], kEpsilon, 1 - kEpsilon));
  }
  return -cross_entropy / normVec1.size();
}

float LInfinityDistance(const std::vector<float> &v1,
                        const std::vector<float> &v2) {
  ASSERT_EQ(v1.size(), v2.size());
  std::vector<float> normVec1 = normalize(v1);
  std::vector<float> normVec2 = normalize(v2);
  float result = 0.0f;
  for (size_t i = 0; i < normVec1.size(); ++i) {
    result += std::max(normVec1[i], normVec2[i]);
  }
  return result;
}

float metric(const std::string &metric_name, const std::vector<float> &v1,
             const std::vector<float> &v2) {
  // currently not specifying config for each metric
  return kMetricCalculators.at(metric_name)(v1, v2);
}
