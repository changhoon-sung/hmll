#ifndef PYHMLL_FORMATTERS_H
#define PYHMLL_FORMATTERS_H

#include <fmt/format.h>
#include "hmll/types.h"

template <>
struct fmt::formatter<hmll_device_t> : formatter<std::string> {
    auto format(const hmll_device_t& device, format_context& ctx) const {
        switch (device) {
        case HMLL_DEVICE_CPU:
            return formatter<std::string>::format("CPU", ctx);
        case HMLL_DEVICE_CUDA:
            return formatter<std::string>::format("CUDA", ctx);
        default:
            return formatter<std::string>::format("unknown", ctx);
        }
    }
};

template <>
struct fmt::formatter<hmll_fetcher_kind_t> : formatter<std::string> {
    auto format(const hmll_fetcher_kind_t& kind, format_context& ctx) const {
        return formatter<std::string>::format("opaque", ctx);
    }
};

#endif // PYHMLL_FORMATTERS_H