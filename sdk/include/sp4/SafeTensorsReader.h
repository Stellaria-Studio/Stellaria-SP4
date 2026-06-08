#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace sp4 {

struct TensorInfo {
    std::string name;
    std::string dtype;
    std::vector<uint32_t> shape;
    uint64_t offsetBegin = 0;
    uint64_t offsetEnd = 0;
};

class SafeTensorsReader {
public:
    bool Open(const std::string& path, std::string* error = nullptr);
    const std::unordered_map<std::string, TensorInfo>& Tensors() const { return tensors_; }
    std::vector<std::string> ConvWeightNames() const;
    std::vector<std::string> SupportedWeightNames() const;
    bool ReadF32(const std::string& name, std::vector<float>& outValues, std::string* error = nullptr) const;

private:
    std::string path_;
    std::vector<uint8_t> payload_;
    std::unordered_map<std::string, TensorInfo> tensors_;
};

} // namespace sp4
