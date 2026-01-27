//
// Integration test for reading safetensors file with multiple dtypes and dimensions
//

#include <catch2/catch_all.hpp>
#include "hmll/hmll.h"
#include <cmath>
#include <complex>

// Fixture path - adjust relative to test binary location
#define SAFETENSORS_TEST_FILE "hmll.safetensors"

// Helper function to validate scalar tensor content
template<typename T>
void validate_scalar_value(const hmll_iobuf_t& buffer, T expected_value) {
    REQUIRE(buffer.ptr != nullptr);
    const T* data = static_cast<const T*>(buffer.ptr);
    REQUIRE(*data == expected_value);
}

// Helper function to validate tensor filled with the same value
template<typename T>
void validate_uniform_tensor(const hmll_iobuf_t& buffer, const size_t num_elements, T expected_value) {
    REQUIRE(buffer.ptr != nullptr);
    const T* data = static_cast<const T*>(buffer.ptr);

    for (size_t i = 0; i < num_elements; ++i) {
        REQUIRE(data[i] == expected_value);
    }
}

// Helper for floating point comparison with tolerance
template<typename T>
void validate_uniform_tensor_approx(const hmll_iobuf_t& buffer, const size_t num_elements, T expected_value, T epsilon = static_cast<T>(1e-5)) {
    REQUIRE(buffer.ptr != nullptr);
    const T* data = static_cast<const T*>(buffer.ptr);

    for (size_t i = 0; i < num_elements; ++i) {
        REQUIRE(std::abs(data[i] - expected_value) < epsilon);
    }
}

// Helper to calculate total elements for N-dimensional tensor with shape [N,N,N,...]
size_t calculate_elements(const uint8_t ndim) {
    if (ndim == 0) return 1;
    size_t total = 1;
    for (uint8_t i = 0; i < ndim; ++i) {
        total *= ndim;
    }
    return total;
}

TEST_CASE("safetensors integration - read multi-dtype file", "[safetensors][integration]")
{
    // Initialize context and source
    hmll_t ctx;
    hmll_source_t src;

    SECTION("can open and parse safetensors file") {
        hmll_error_t err = hmll_source_open(SAFETENSORS_TEST_FILE, &src);
        REQUIRE(hmll_success(err));

        // Parse registry
        hmll_registry_t registry = {};
        size_t num_tensors = hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0);
        REQUIRE(num_tensors > 0);
        REQUIRE(registry.num_tensors == num_tensors);

        // Initialize loader
        err = hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, HMLL_FETCHER_MMAP);
        REQUIRE(hmll_success(err));

        INFO("Successfully loaded " << num_tensors << " tensors from safetensors file");

        // Cleanup
        hmll_free_registry(&registry);
        hmll_destroy(&ctx);
        hmll_source_close(&src);
    }

    SECTION("validate float32 tensors across dimensions") {
        REQUIRE(hmll_success(hmll_source_open(SAFETENSORS_TEST_FILE, &src)));

        hmll_registry_t registry = {};
        REQUIRE(hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0) > 0);
        REQUIRE(hmll_success(hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, HMLL_FETCHER_MMAP)));

        // Test dimensions 0 through 5
        for (uint8_t ndim = 0; ndim <= 5; ++ndim) {
            std::string tensor_name = "float32.dim" + std::to_string(ndim);
            INFO("Testing tensor: " << tensor_name);

            const hmll_lookup_result_t lookup = hmll_lookup_tensor(&ctx, &registry, tensor_name.c_str());
            REQUIRE(hmll_success(ctx.error));
            REQUIRE(lookup.specs != nullptr);
            REQUIRE(lookup.specs->dtype == HMLL_DTYPE_FLOAT32);
            REQUIRE(lookup.specs->rank == ndim);

            // Validate shape
            for (uint8_t i = 0; i < ndim; ++i) {
                REQUIRE(lookup.specs->shape[i] == ndim);
            }

            // Fetch tensor data
            const hmll_range_t range = {lookup.specs->start, lookup.specs->end};
            const hmll_iobuf_t buffer = hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, range);
            REQUIRE(hmll_success(ctx.error));

            ssize_t bytes_read = hmll_fetch(&ctx, lookup.file, &buffer, range);
            REQUIRE(bytes_read > 0);
            REQUIRE(hmll_success(ctx.error));

            // Validate content: dim N should contain value N
            size_t num_elements = calculate_elements(ndim);
            auto expected_value = static_cast<float>(ndim);
            validate_uniform_tensor_approx<float>(buffer, num_elements, expected_value);

            hmll_free_buffer(const_cast<hmll_iobuf_t*>(&buffer));
        }

        hmll_free_registry(&registry);
        hmll_destroy(&ctx);
        hmll_source_close(&src);
    }

    SECTION("validate int32 tensors across dimensions") {
        REQUIRE(hmll_success(hmll_source_open(SAFETENSORS_TEST_FILE, &src)));

        hmll_registry_t registry = {};
        REQUIRE(hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0) > 0);
        REQUIRE(hmll_success(hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, HMLL_FETCHER_MMAP)));

        for (uint8_t ndim = 0; ndim <= 5; ++ndim) {
            std::string tensor_name = "int32.dim" + std::to_string(ndim);
            INFO("Testing tensor: " << tensor_name);

            const hmll_lookup_result_t lookup = hmll_lookup_tensor(&ctx, &registry, tensor_name.c_str());
            REQUIRE(hmll_success(ctx.error));
            REQUIRE(lookup.specs != nullptr);
            REQUIRE(lookup.specs->dtype == HMLL_DTYPE_SIGNED_INT32);
            REQUIRE(lookup.specs->rank == ndim);

            const hmll_range_t range = {lookup.specs->start, lookup.specs->end};
            const hmll_iobuf_t buffer = hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, range);
            REQUIRE(hmll_success(ctx.error));

            ssize_t bytes_read = hmll_fetch(&ctx, lookup.file, &buffer, range);
            REQUIRE(bytes_read > 0);

            size_t num_elements = calculate_elements(ndim);
            auto expected_value = static_cast<int32_t>(ndim);
            validate_uniform_tensor<int32_t>(buffer, num_elements, expected_value);

            hmll_free_buffer(const_cast<hmll_iobuf_t*>(&buffer));
        }

        hmll_free_registry(&registry);
        hmll_destroy(&ctx);
        hmll_source_close(&src);
    }

    SECTION("validate uint8 tensors across dimensions") {
        REQUIRE(hmll_success(hmll_source_open(SAFETENSORS_TEST_FILE, &src)));

        hmll_registry_t registry = {};
        REQUIRE(hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0) > 0);
        REQUIRE(hmll_success(hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, HMLL_FETCHER_MMAP)));

        for (uint8_t ndim = 0; ndim <= 5; ++ndim) {
            std::string tensor_name = "uint8.dim" + std::to_string(ndim);
            INFO("Testing tensor: " << tensor_name);

            const hmll_lookup_result_t lookup = hmll_lookup_tensor(&ctx, &registry, tensor_name.c_str());
            REQUIRE(hmll_success(ctx.error));
            REQUIRE(lookup.specs != nullptr);
            REQUIRE(lookup.specs->dtype == HMLL_DTYPE_UNSIGNED_INT8);
            REQUIRE(lookup.specs->rank == ndim);

            const hmll_range_t range = {lookup.specs->start, lookup.specs->end};
            const hmll_iobuf_t buffer = hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, range);
            REQUIRE(hmll_success(ctx.error));

            ssize_t bytes_read = hmll_fetch(&ctx, lookup.file, &buffer, range);
            REQUIRE(bytes_read > 0);

            size_t num_elements = calculate_elements(ndim);
            auto expected_value = static_cast<uint8_t>(ndim);
            validate_uniform_tensor<uint8_t>(buffer, num_elements, expected_value);

            hmll_free_buffer(const_cast<hmll_iobuf_t*>(&buffer));
        }

        hmll_free_registry(&registry);
        hmll_destroy(&ctx);
        hmll_source_close(&src);
    }

    SECTION("validate int64 tensors across dimensions") {
        REQUIRE(hmll_success(hmll_source_open(SAFETENSORS_TEST_FILE, &src)));

        hmll_registry_t registry = {};
        REQUIRE(hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0) > 0);
        REQUIRE(hmll_success(hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, HMLL_FETCHER_MMAP)));

        for (uint8_t ndim = 0; ndim <= 5; ++ndim) {
            std::string tensor_name = "int64.dim" + std::to_string(ndim);
            INFO("Testing tensor: " << tensor_name);

            const hmll_lookup_result_t lookup = hmll_lookup_tensor(&ctx, &registry, tensor_name.c_str());
            REQUIRE(hmll_success(ctx.error));
            REQUIRE(lookup.specs != nullptr);
            REQUIRE(lookup.specs->dtype == HMLL_DTYPE_SIGNED_INT64);
            REQUIRE(lookup.specs->rank == ndim);

            const hmll_range_t range = {lookup.specs->start, lookup.specs->end};
            const hmll_iobuf_t buffer = hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, range);
            REQUIRE(hmll_success(ctx.error));

            ssize_t bytes_read = hmll_fetch(&ctx, lookup.file, &buffer, range);
            REQUIRE(bytes_read > 0);

            size_t num_elements = calculate_elements(ndim);
            auto expected_value = static_cast<int64_t>(ndim);
            validate_uniform_tensor<int64_t>(buffer, num_elements, expected_value);

            hmll_free_buffer(const_cast<hmll_iobuf_t*>(&buffer));
        }

        hmll_free_registry(&registry);
        hmll_destroy(&ctx);
        hmll_source_close(&src);
    }

    SECTION("validate bfloat16 tensors exist and can be fetched") {
        REQUIRE(hmll_success(hmll_source_open(SAFETENSORS_TEST_FILE, &src)));

        hmll_registry_t registry = {};
        REQUIRE(hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0) > 0);
        REQUIRE(hmll_success(hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, HMLL_FETCHER_MMAP)));

        for (uint8_t ndim = 0; ndim <= 5; ++ndim) {
            std::string tensor_name = "bfloat16.dim" + std::to_string(ndim);
            INFO("Testing tensor: " << tensor_name);

            const hmll_lookup_result_t lookup = hmll_lookup_tensor(&ctx, &registry, tensor_name.c_str());
            REQUIRE(hmll_success(ctx.error));
            REQUIRE(lookup.specs != nullptr);
            REQUIRE(lookup.specs->dtype == HMLL_DTYPE_BFLOAT16);
            REQUIRE(lookup.specs->rank == ndim);

            const hmll_range_t range = {lookup.specs->start, lookup.specs->end};
            const hmll_iobuf_t buffer = hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, range);
            REQUIRE(hmll_success(ctx.error));

            ssize_t bytes_read = hmll_fetch(&ctx, lookup.file, &buffer, range);
            REQUIRE(bytes_read > 0);
            REQUIRE(hmll_success(ctx.error));

            hmll_free_buffer(const_cast<hmll_iobuf_t*>(&buffer));
        }

        hmll_free_registry(&registry);
        hmll_destroy(&ctx);
        hmll_source_close(&src);
    }

    SECTION("validate complex64 tensors exist") {
        REQUIRE(hmll_success(hmll_source_open(SAFETENSORS_TEST_FILE, &src)));

        hmll_registry_t registry = {};
        REQUIRE(hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0) > 0);
        REQUIRE(hmll_success(hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, HMLL_FETCHER_MMAP)));

        for (uint8_t ndim = 0; ndim <= 5; ++ndim) {
            std::string tensor_name = "complex64.dim" + std::to_string(ndim);
            INFO("Testing tensor: " << tensor_name);

            const hmll_lookup_result_t lookup = hmll_lookup_tensor(&ctx, &registry, tensor_name.c_str());
            REQUIRE(hmll_success(ctx.error));
            REQUIRE(lookup.specs != nullptr);
            REQUIRE(lookup.specs->dtype == HMLL_DTYPE_COMPLEX);
            REQUIRE(lookup.specs->rank == ndim);

            const hmll_range_t range = {lookup.specs->start, lookup.specs->end};
            const hmll_iobuf_t buffer = hmll_get_buffer_for_range(&ctx, ctx.fetcher->device, range);
            REQUIRE(hmll_success(ctx.error));

            ssize_t bytes_read = hmll_fetch(&ctx, lookup.file, &buffer, range);
            REQUIRE(bytes_read > 0);

            hmll_free_buffer(const_cast<hmll_iobuf_t*>(&buffer));
        }

        hmll_free_registry(&registry);
        hmll_destroy(&ctx);
        hmll_source_close(&src);
    }

    SECTION("validate all expected tensor names exist") {
        REQUIRE(hmll_success(hmll_source_open(SAFETENSORS_TEST_FILE, &src)));

        hmll_registry_t registry = {};
        REQUIRE(hmll_safetensors_populate_registry(&ctx, &registry, src, 0, 0) > 0);
        REQUIRE(hmll_success(hmll_loader_init(&ctx, &src, 1, HMLL_DEVICE_CPU, HMLL_FETCHER_MMAP)));

        // Test a sampling of expected dtype+dimension combinations
        std::vector<std::string> expected_tensors = {
            "float32.dim0", "float32.dim1", "float32.dim5",
            "int32.dim0", "int32.dim3",
            "uint8.dim0", "uint8.dim4",
            "int64.dim2",
            "bfloat16.dim1",
            "float16.dim0",
            "complex64.dim0",
            "float8_e4m3fn.dim0",
            "float8_e5m2.dim0"
        };

        for (const auto& tensor_name : expected_tensors) {
            INFO("Checking for tensor: " << tensor_name);
            unsigned char exists = hmll_contains(&ctx, &registry, tensor_name.c_str());
            REQUIRE(exists == 1);
        }

        hmll_free_registry(&registry);
        hmll_destroy(&ctx);
        hmll_source_close(&src);
    }
}
