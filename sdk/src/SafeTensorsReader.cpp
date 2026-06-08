#include "sp4/SafeTensorsReader.h"

#include "SP4Common.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <regex>
#include <sstream>

namespace sp4 {

namespace {

std::vector<uint32_t> ParseShape(const std::string& text) {
    std::vector<uint32_t> shape;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char c) { return std::isspace(c) != 0; }), item.end());
        if (!item.empty()) {
            shape.push_back(static_cast<uint32_t>(std::stoul(item)));
        }
    }
    return shape;
}

bool IsReadableFloat(const std::string& dtype) {
    return dtype == "F32" || dtype == "F16" || dtype == "BF16";
}

float BFloat16ToFloat(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16u;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
}

} // namespace

bool SafeTensorsReader::Open(const std::string& path, std::string* error) {
    path_ = path;
    tensors_.clear();
    payload_.clear();

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        if (error) *error = "failed to open safetensors: " + path;
        return false;
    }
    const std::streamsize size = file.tellg();
    if (size < 16) {
        if (error) *error = "safetensors file is too small";
        return false;
    }
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> raw(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(raw.data()), size);
    if (!file) {
        if (error) *error = "failed to read safetensors";
        return false;
    }

    const uint64_t headerBytes = ReadU64(raw.data());
    if (8u + headerBytes > raw.size()) {
        if (error) *error = "invalid safetensors header length";
        return false;
    }
    const std::string header(reinterpret_cast<const char*>(raw.data() + 8), static_cast<size_t>(headerBytes));
    payload_.assign(raw.begin() + 8 + static_cast<size_t>(headerBytes), raw.end());

    const std::regex tensorRegex("\"([^\"]+)\"\\s*:\\s*\\{\\s*\"dtype\"\\s*:\\s*\"([^\"]+)\"\\s*,\\s*\"shape\"\\s*:\\s*\\[([^\\]]*)\\]\\s*,\\s*\"data_offsets\"\\s*:\\s*\\[([0-9]+)\\s*,\\s*([0-9]+)\\]");
    for (auto it = std::sregex_iterator(header.begin(), header.end(), tensorRegex);
         it != std::sregex_iterator();
         ++it) {
        TensorInfo info;
        info.name = (*it)[1].str();
        info.dtype = (*it)[2].str();
        info.shape = ParseShape((*it)[3].str());
        info.offsetBegin = static_cast<uint64_t>(std::stoull((*it)[4].str()));
        info.offsetEnd = static_cast<uint64_t>(std::stoull((*it)[5].str()));
        if (info.name != "__metadata__") {
            tensors_[info.name] = std::move(info);
        }
    }

    if (tensors_.empty()) {
        if (error) *error = "no tensors parsed from safetensors header";
        return false;
    }
    return true;
}

bool SafeTensorsReader::ReadF32(const std::string& name, std::vector<float>& outValues, std::string* error) const {
    outValues.clear();
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        if (error) *error = "missing tensor: " + name;
        return false;
    }
    const TensorInfo& info = it->second;
    if (!IsReadableFloat(info.dtype)) {
        if (error) *error = "tensor is not F32/F16/BF16: " + name;
        return false;
    }
    if (info.offsetEnd < info.offsetBegin || info.offsetEnd > payload_.size()) {
        if (error) *error = "tensor storage range invalid: " + name;
        return false;
    }
    uint64_t count = 1;
    for (uint32_t dim : info.shape) {
        count *= dim;
    }
    const uint64_t elementBytes = info.dtype == "F32" ? sizeof(float) : sizeof(uint16_t);
    if ((info.offsetEnd - info.offsetBegin) != count * elementBytes) {
        if (error) *error = "tensor byte count mismatch: " + name;
        return false;
    }
    outValues.resize(static_cast<size_t>(count));
    if (info.dtype == "F32") {
        std::memcpy(outValues.data(), payload_.data() + info.offsetBegin, static_cast<size_t>(count * sizeof(float)));
    } else {
        for (size_t i = 0; i < outValues.size(); ++i) {
            uint16_t raw = 0;
            std::memcpy(&raw, payload_.data() + info.offsetBegin + i * sizeof(uint16_t), sizeof(uint16_t));
            outValues[i] = info.dtype == "F16" ? HalfToFloat(raw) : BFloat16ToFloat(raw);
        }
    }
    return true;
}

std::vector<std::string> SafeTensorsReader::ConvWeightNames() const {
    std::vector<std::string> weights;
    for (const auto& [name, info] : tensors_) {
        if (IsReadableFloat(info.dtype) && info.shape.size() == 4 && name.size() >= 7 && name.rfind(".weight") == name.size() - 7) {
            weights.push_back(name);
        }
    }
    std::sort(weights.begin(), weights.end());
    std::vector<std::string> ordered;
    for (int i = 0; i < 3; ++i) {
        const std::string first = "block" + std::to_string(i) + ".conv0.0.0.weight";
        if (std::find(weights.begin(), weights.end(), first) != weights.end()) {
            ordered.push_back(first);
        }
    }
    for (const std::string& name : weights) {
        if (std::find(ordered.begin(), ordered.end(), name) == ordered.end()) {
            ordered.push_back(name);
        }
    }
    return ordered;
}

std::vector<std::string> SafeTensorsReader::SupportedWeightNames() const {
    std::vector<std::string> weights;
    for (const auto& [name, info] : tensors_) {
        const bool isWeight = name.size() >= 7 && name.rfind(".weight") == name.size() - 7;
        const bool supportedShape = info.shape.size() == 2 || info.shape.size() == 4;
        if (IsReadableFloat(info.dtype) && isWeight && supportedShape) {
            weights.push_back(name);
        }
    }
    std::sort(weights.begin(), weights.end());
    std::vector<std::string> ordered;
    for (int i = 0; i < 3; ++i) {
        const std::string first = "block" + std::to_string(i) + ".conv0.0.0.weight";
        if (std::find(weights.begin(), weights.end(), first) != weights.end()) {
            ordered.push_back(first);
        }
    }
    for (const std::string& name : weights) {
        if (std::find(ordered.begin(), ordered.end(), name) == ordered.end()) {
            ordered.push_back(name);
        }
    }
    return ordered;
}

} // namespace sp4
