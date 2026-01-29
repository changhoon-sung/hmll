#ifndef PYHMLL_FORMATTERS_H
#define PYHMLL_FORMATTERS_H

#include <fmt/format.h>
#include "hmll/types.h"

constexpr auto format_as(const hmll_device_t f) { return fmt::underlying(f); }
constexpr auto format_as(const hmll_fetcher_kind_t f) { return fmt::underlying(f); }

#endif // PYHMLL_FORMATTERS_H