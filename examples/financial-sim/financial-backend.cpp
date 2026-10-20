#include "financial-backend.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

static std::string uppercase(const char * value) {
    std::string result = value ? value : "";
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return result;
}

static int backend_priority(ggml_backend_dev_t device) {
    const std::string name = uppercase(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device))) +
                             uppercase(ggml_backend_dev_name(device));
    if (name.find("FPGA") != std::string::npos) {
        return 0;
    }
    if (name.find("CUDA") != std::string::npos) {
        return 1;
    }
    if (name.find("METAL") != std::string::npos) {
        return 2;
    }

    switch (ggml_backend_dev_type(device)) {
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return 3;
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return 4;
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return 5;
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return 6;
    }
    return 7;
}

ggml_backend_dev_t financial_backend_select(const char * name) {
    const std::string requested = uppercase(name);
    const bool automatic = requested.empty() || requested == "AUTO";
    std::vector<ggml_backend_dev_t> devices;

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const std::string device_name = uppercase(ggml_backend_dev_name(device));
        const std::string backend_name = uppercase(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)));
        if (!automatic && requested != device_name && requested != backend_name) {
            continue;
        }
        devices.push_back(device);
    }

    if (devices.empty()) {
        return nullptr;
    }

    return *std::min_element(devices.begin(), devices.end(), [](ggml_backend_dev_t left, ggml_backend_dev_t right) {
        return backend_priority(left) < backend_priority(right);
    });
}
