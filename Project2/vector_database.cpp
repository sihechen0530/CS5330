#include "vector_database.h"
#include "csv_util.h"
#include "feature_extractor.h"
#include "metric_calculator.h"
#include "utils.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>

const std::vector<std::string> kImageSuffix = {".jpg", ".png", ".ppm", ".tif"};

int read_image_data_csv(const std::string &file_path,
                        std::vector<std::string> *image_paths,
                        std::vector<std::vector<float>> *feature_map) {
  std::vector<char *> image_path_ptrs;
  if (0 != read_image_data_csv(file_path.c_str(), image_path_ptrs, *feature_map,
                               0)) {
    LOG_ERROR("failed to read csv file " << file_path);
    return -1;
  }
  for (const auto &image_path_ptr : image_path_ptrs) {
    image_paths->emplace_back(image_path_ptr);
  }
  return 0;
}

int dump_image_data_csv(const std::string &file_path,
                        const std::vector<std::string> &image_paths,
                        const std::vector<std::vector<float>> &feature_map,
                        const std::vector<bool> &status) {
  ASSERT_EQ(image_paths.size(), feature_map.size());
  // clear file before dumping
  std::ofstream file(file_path, std::ios::out);
  if (file.is_open()) {
    file.close();
  } else {
    LOG_ERROR("Error opening file.");
  }
  int n = image_paths.size();
  for (size_t i = 0; i < image_paths.size(); ++i) {
    // only dump the valid features
    if (status[i] &&
        0 != append_image_data_csv(file_path.c_str(), image_paths[i].c_str(),
                                   feature_map[i])) {
      LOG_ERROR("failed to append data to csv file");
      return -1;
    }
  }
  return 0;
}

std::string feature_file_path_formatter(const std::string &directory,
                                        const std::string &feature_type) {
  return directory + "/" + feature_type + ".csv";
}

VectorDatabase::VectorDatabase(const std::string &directory) {
  // step 1: set directory
  directory_ = directory;

  // step 2: load all image paths
  DIR *dirp;
  struct dirent *dp;
  dirp = opendir(directory_.c_str());
  if (dirp == NULL) {
    LOG_ERROR("Cannot open directory: " << directory_);
    exit(-1);
  }
  while ((dp = readdir(dirp)) != NULL) {
    bool is_image = false;
    for (const auto &suffix : kImageSuffix) {
      if (strstr(dp->d_name, suffix.c_str())) {
        is_image = true;
        break;
      }
    }
    if (is_image) {
      image_paths_.emplace_back(dp->d_name);
    }
  }
  std::sort(image_paths_.begin(), image_paths_.end());
  for (size_t i = 0; i < image_paths_.size(); ++i) {
    image_2_index_[image_paths_[i]] = i;
  }
}

int VectorDatabase::loadFeatureByType(const std::string &feature_type) {
  // 1. check if feature is loaded
  if (database_.find(feature_type) != database_.end()) {
    LOG_INFO("feature is already loaded " << feature_type);
    return 0;
  }
  // 2. init feature database
  database_[feature_type].resize(image_paths_.size());
  status_[feature_type] = std::vector<bool>(image_paths_.size(), false);
  // 3. load feature from file
  // feature_type is a string. add suffix with ".csv" to become a file
  const std::string feature_file =
      feature_file_path_formatter(directory_, feature_type);
  std::vector<std::string> feature_keys;
  std::vector<std::vector<float>> feature_maps;
  if (0 != read_image_data_csv(feature_file, &feature_keys, &feature_maps)) {
    LOG_ERROR("failed to load feature from file " << feature_file);
    return -1;
  }
  // 4. keep a copy in database
  ASSERT_EQ(feature_keys.size(), database_[feature_type].size());
  for (size_t i = 0; i < feature_keys.size(); ++i) {
    database_[feature_type][image_2_index_[feature_keys[i]]] = feature_maps[i];
    status_[feature_type][image_2_index_[feature_keys[i]]] = true;
  }
  return 0;
}

int VectorDatabase::save() const {
  for (const auto &data : database_) {
    const auto &feature_type = data.first;
    const auto &feature_maps = data.second;
    const auto &status = status_.at(feature_type);
    if (0 != dump_image_data_csv(
                 feature_file_path_formatter(directory_, feature_type),
                 image_paths_, feature_maps, status)) {
      LOG_ERROR("failed to dump feature to file " << feature_type);
    }
  }
  return 0;
}

int VectorDatabase::getFeature(const std::string &feature_type,
                               const std::string &image_path,
                               std::vector<float> *feature) const {
  if (database_.find(feature_type) == database_.end() ||
      !status_.at(feature_type).at(image_2_index_.at(image_path))) {
    LOG_ERROR("no such feature " << feature_type);
    return -1;
  }
  *feature = database_.at(feature_type).at(image_2_index_.at(image_path));
  return 0;
}

int VectorDatabase::updateFeature(const std::string &feature_type,
                                  const nlohmann::json &config) {
  // TODO: omp parallelize
  // #pragma omp parallel for
  for (const auto &image_path : image_paths_) {
    size_t index = image_2_index_[image_path];
    if (status_[feature_type][index]) {
      // only process new images, skip.
      continue;
    }
    const auto &realpath = directory_ + "/" + image_path;
    if (0 != extract(realpath, config, &database_[feature_type][index])) {
      LOG_ERROR("failed to compute feature " << feature_type << " "
                                             << image_path);
      return -1;
    }
    status_[feature_type][index] = true;
  }
  return 0;
}

int VectorDatabase::getTopKMatch(const std::string &feature_type,
                                 const std::string &metric_name,
                                 const std::string &target_path, int k,
                                 std::vector<std::string> *matches) const {
  // step 1: check parameters
  if (image_2_index_.find(target_path) == image_2_index_.end()) {
    LOG_ERROR("image not found " << target_path);
    return -1;
  }
  if (database_.find(feature_type) == database_.end()) {
    LOG_ERROR("feature type not founded in database");
    return -1;
  }
  if (!status_.at(feature_type).at(image_2_index_.at(target_path))) {
    LOG_ERROR("feature not ready " << target_path << " " << feature_type);
    return -1;
  }
  if (k >= image_paths_.size()) {
    LOG_WARN("k is greater than image_paths");
    *matches = image_paths_;
    return 0;
  }
  matches->clear();
  matches->reserve(k);
  // step 2: calculate metric value
  // TODO: omp parallelize
  std::vector<std::tuple<float, size_t>> metric_values;
  metric_values.reserve(image_paths_.size());
  const auto &target_vector =
      database_.at(feature_type).at(image_2_index_.at(target_path));
  for (size_t i = 0; i < image_paths_.size(); ++i) {
    const auto &compare_vector = database_.at(feature_type).at(i);
    metric_values.emplace_back(
        metric(metric_name, target_vector, compare_vector), i);
  }
  // step 3: get top k
  std::nth_element(metric_values.begin(), metric_values.begin() + k,
                   metric_values.end(), [&](const auto &lhs, const auto &rhs) {
                     return std::get<0>(lhs) < std::get<0>(rhs);
                   });
  // step 4: return matches
  for (auto iter = metric_values.begin(); iter != metric_values.begin() + k;
       ++iter) {
    const auto &image_path = image_paths_.at(std::get<1>(*iter));
    LOG_INFO(image_path << " score: " << std::get<0>(*iter));
    matches->emplace_back(image_path);
  }
  return 0;
}
