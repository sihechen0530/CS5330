#ifndef VECTOR_DATABASE_H
#define VECTOR_DATABASE_H

#include <functional>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

class VectorDatabase {
public:
  // constructors
  VectorDatabase() = default;
  VectorDatabase(const std::string &directory);
  ~VectorDatabase() = default;

  // load feature by feature type string. the file path is
  // <directory_>/<feature_type>.csv
  int loadFeatureByType(const std::string &feature_type);

  // save all feature vectors to files. (called on exit)
  int save() const;

  // get specific type of feature vector of image
  int getFeature(const std::string &feature_type, const std::string &image_path,
                 std::vector<float> *feature) const;

  // update a specific type of feature with config
  int updateFeature(const std::string &feature_type,
                    const nlohmann::json &config);

  // return topk matches of a image of specific feature type using specific
  // metrics
  int getTopKMatch(const std::string &feature_type,
                   const std::string &metric_name,
                   const std::string &target_path, int k,
                   std::vector<std::string> *matches) const;

private:
  // base directory
  std::string directory_;
  // all image paths in the directory (sorted to match the order in database)
  std::vector<std::string> image_paths_;
  // map image path to index in database
  std::unordered_map<std::string, size_t> image_2_index_;
  // feature_type -> feature vector in the order of image path
  std::unordered_map<std::string, std::vector<std::vector<float>>> database_;
  // whether an image's feature is valid or not
  std::unordered_map<std::string, std::vector<bool>> status_;
};

#endif
