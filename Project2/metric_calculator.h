#ifndef METRIC_CALCULATOR_H
#define METRIC_CALCULATOR_H
#include <string>
#include <vector>

float metric(const std::string &metric_name, const std::vector<float> &v1,
             const std::vector<float> &v2);

// all matchers only accept std::vector<float> as input
// return value: success 0 otherwise 1
float sumOfSquaredDifference(const std::vector<float> &v1,
                             const std::vector<float> &v2);

float histogramIntersection(const std::vector<float> &v1,
                            const std::vector<float> &v2);

float cosine(const std::vector<float> &v1, const std::vector<float> &v2);

#endif
