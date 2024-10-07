#ifndef METRIC_CALCULATOR_H
#define METRIC_CALCULATOR_H
#include <string>
#include <vector>

/**
 * The external entrance of all metrics methods. Input the metric name and two
 * feature vectors of float values and compute the metric with designated method
 * and return the metric as a single float (smaller meaning more similarity).
 *
 * This function takes two float vectors and return a float.
 *
 * @param metric_name a string of metric name (registered in std::unordered_map
 * in cpp file)
 * @param v1 feature vector 1
 * @param v2 feature vector 2
 * @return The metric value
 */
float metric(const std::string &metric_name, const std::vector<float> &v1,
             const std::vector<float> &v2);

/**
 * sum of squared difference implementation. \sum (v1[i] - v2[i])^2
 *
 * @param v1 feature vector 1
 * @param v2 feature vector 2
 * @return The metric value
 */
float sumOfSquaredDifference(const std::vector<float> &v1,
                             const std::vector<float> &v2);

/**
 * histogram intersection. \sum min(v1[i], v2[i])
 *
 * @param v1 feature vector 1
 * @param v2 feature vector 2
 * @return The metric value
 */
float histogramIntersection(const std::vector<float> &v1,
                            const std::vector<float> &v2);

/**
 * cosine distance. \frac {v1 \cdot v2}{||v1||||v2||}
 *
 * @param v1 feature vector 1
 * @param v2 feature vector 2
 * @return The metric value
 */
float cosine(const std::vector<float> &v1, const std::vector<float> &v2);

/**
 * cross entropy. -\sum {v1[i] * log(v2[i])}
 *
 * @param v1 feature vector 1
 * @param v2 feature vector 2
 * @return The metric value
 */
float crossEntropy(const std::vector<float> &v1, const std::vector<float> &v2);

/**
 * LN distance. \sum max(v1[i], v2[i])
 *
 * @param v1 feature vector 1
 * @param v2 feature vector 2
 * @return The metric value
 */
float LInfinityDistance(const std::vector<float> &v1,
                        const std::vector<float> &v2);

#endif
