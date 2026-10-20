#include "financial-backend.h"

#include <cstring>
#include <iostream>

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t cpu = financial_backend_select("cpu");
    if (!cpu || std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(cpu)), "CPU") != 0) {
        std::cerr << "CPU backend selection failed\n";
        return 1;
    }

    ggml_backend_dev_t automatic = financial_backend_select("auto");
    if (!automatic) {
        std::cerr << "Automatic backend selection failed\n";
        return 1;
    }

    if (financial_backend_select("unavailable-backend") != nullptr) {
        std::cerr << "Unavailable backend should not be selected\n";
        return 1;
    }

    std::cout << "Financial backend selection tests passed\n";
    return 0;
}
