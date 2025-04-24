#ifndef DATALOADER_H
#define DATALOADER_H

#include <vector>
#include <string>

// Function prototypes
void load_csv_wdbc(const std::string& filename, std::vector<std::vector<float>>& features, std::vector<float>& labels);
void load_csv_votes(const std::string& filename, std::vector<std::vector<float>>& features, std::vector<float>& labels);
void load_csv_iris(const std::string& filename, std::vector<std::vector<float>>& features, std::vector<float>& labels);

#endif 