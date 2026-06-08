#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace sp4 {

inline void AppendU32(std::vector<uint8_t>& out, uint32_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xffu));
    out.push_back(static_cast<uint8_t>((value >> 8u) & 0xffu));
    out.push_back(static_cast<uint8_t>((value >> 16u) & 0xffu));
    out.push_back(static_cast<uint8_t>((value >> 24u) & 0xffu));
}

inline uint32_t ReadU32(const uint8_t* data) {
    return static_cast<uint32_t>(data[0]) |
           (static_cast<uint32_t>(data[1]) << 8u) |
           (static_cast<uint32_t>(data[2]) << 16u) |
           (static_cast<uint32_t>(data[3]) << 24u);
}

inline uint64_t ReadU64(const uint8_t* data) {
    uint64_t value = 0;
    for (int i = 7; i >= 0; --i) {
        value = (value << 8u) | data[i];
    }
    return value;
}

inline uint16_t FloatToHalf(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16u) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23u) & 0xffu) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa = (mantissa | 0x800000u) >> static_cast<uint32_t>(1 - exp);
        return static_cast<uint16_t>(sign | ((mantissa + 0x1000u) >> 13u));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10u) | ((mantissa + 0x1000u) >> 13u));
}

inline float HalfToFloat(uint16_t half) {
    const uint32_t sign = (static_cast<uint32_t>(half & 0x8000u)) << 16u;
    uint32_t exp = (half >> 10u) & 0x1fu;
    uint32_t mantissa = half & 0x03ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1u;
                --exp;
            }
            mantissa &= 0x03ffu;
            bits = sign | ((exp + 127u - 15u) << 23u) | (mantissa << 13u);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13u);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23u) | (mantissa << 13u);
    }
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline std::string JsonEscape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (char c : text) {
        if (c == '"' || c == '\\') {
            out.push_back('\\');
            out.push_back(c);
        } else if (c == '\n') {
            out += "\\n";
        } else {
            out.push_back(c);
        }
    }
    return out;
}

inline std::string BasenameNoExt(const std::string& name) {
    const size_t dot = name.rfind(".weight");
    return dot == std::string::npos ? name : name.substr(0, dot);
}

} // namespace sp4
