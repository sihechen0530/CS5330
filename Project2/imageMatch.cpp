/*
 */
#include "utils.h"
#include "vector_database.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

const std::string kFeatureTypeKey = "feature_type";
const std::string kFeatureExtractorsKey = "feature_extractors";
const std::string kMetricKey = "metric";
const std::string kTargetImageKey = "target_image";
const std::string kTopK = "K";

constexpr char kQuitKey = 'q';
constexpr char kNextKey = 'n';

int on_exit(VectorDatabase &vector_database) {
  vector_database.save();
  return 0;
}

int main(const int argc, const char *argv[]) {
  // check for sufficient arguments
  if (argc < 3) {
    printf("usage: %s <directory path> <config_path>\n", argv[0]);
    exit(-1);
  }

  // 1. init database
  std::string dirname(argv[1]);
  VectorDatabase vector_database(dirname);
  // 2. load config file
  std::string config_path(argv[2]);
  std::ifstream file(config_path);
  // Check if the file was opened successfully
  if (!file.is_open()) {
    LOG_ERROR("Error: Could not open the file." << config_path);
    return 1;
  }
  // Parse the JSON file
  nlohmann::json config;
  file >> config;
  // Close the file
  file.close();
  // 3. perform tasks
  for (const auto &[task_name, task_cfg] : config.items()) {
    const std::string &feature_type = task_cfg[kFeatureTypeKey];
    if (0 != vector_database.loadFeatureByType(feature_type)) {
      LOG_INFO("failed to load feature. start computing feature type "
               << feature_type);
      if (0 != vector_database.updateFeature(feature_type,
                                             task_cfg[kFeatureExtractorsKey])) {
        LOG_ERROR("failed to compute feature " << feature_type);
        continue;
      };
    }

    std::vector<std::string> matches;
    if (0 != vector_database.getTopKMatch(feature_type, task_cfg[kMetricKey],
                                          task_cfg[kTargetImageKey],
                                          task_cfg[kTopK], &matches)) {
      LOG_ERROR("failed to get top k matches");
      continue;
    };
    showMatches(dirname, matches);
    LOG_INFO("Press n for next task; press q quit");
    while (true) {
      int key = cv::waitKey(0);

      if (kNextKey == key) {
        break;
      }

      if (kQuitKey == key) {
        return on_exit(vector_database);
      }
    }
    cv::destroyAllWindows();
  }
  return on_exit(vector_database);
}
