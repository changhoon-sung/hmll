#include <iostream>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

void init_loader(nb::module_&);

#ifdef __HMLL_SAFETENSORS_ENABLED__
void init_safetensors(nb::module_&);
void init_specs(nb::module_&);
#endif

NB_MODULE(_pyhmll_impl, m)
{
    m.doc() = "hmll: High-Performance Model Loading Library - Efficient AI Model loading for modern AI workloads.";

    init_loader(m);

#ifdef __HMLL_SAFETENSORS_ENABLED__
    init_safetensors(m);
    init_specs(m);
#endif
}