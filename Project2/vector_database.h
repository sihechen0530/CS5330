#ifndef VECTOR_DATABASE_H
#define VECTOR_DATABASE_H

#include <functional>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

class VectorDatabase {
public:
  VectorDatabase(const std::string &directory) {
    directory_ = directory;
    initialize();
  }
  int loadFeatureByType(const std::string &feature_type);
  int save() const;
  int getFeature(const std::string &feature_type, const std::string &image_path,
                 std::vector<float> *feature) const;
  int updateFeature(const std::string &feature_type,
                    const nlohmann::json &config);
  int getTopKMatch(const std::string &feature_type,
                   const std::string &metric_name,
                   const std::string &target_path, int k,
                   std::vector<std::string> *matches) const;

private:
  // sort image path!!
  int initialize();

private:
  std::string directory_;
  std::vector<std::string> image_paths_;
  std::unordered_map<std::string, size_t> image_2_index_;
  // feature_type -> feature vector in the order of image path
  std::unordered_map<std::string, std::vector<std::vector<float>>> database_;
  std::unordered_map<std::string, std::vector<bool>> status_;
};

#endif
