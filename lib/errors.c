//
// Created by mfuntowicz on 12/11/25.
//
#include <string.h>

#include "hmll/hmll.h"
#include "hmll/types.h"

#if defined(_WIN32)
// Thread-local buffer for strerror on Windows
static __declspec(thread) char strerror_buf[256];
#endif

const char *hmll_strerr(const struct hmll_error err)
{
    if (hmll_error_is_os_error(err)) {
#if defined(_WIN32)
        if (strerror_s(strerror_buf, sizeof(strerror_buf), err.sys_err) == 0)
            return strerror_buf;
        return "Unknown system error";
#else
        return strerror(err.sys_err);
#endif
    }

    if (hmll_error_is_lib_error(err))
    {
        switch (err.code)
        {
        case HMLL_ERR_SUCCESS: return "All good";
        case HMLL_ERR_FILE_NOT_FOUND: return "File not found";
        case HMLL_ERR_ALLOCATION_FAILED: return "Failed to allocate memory";
        case HMLL_ERR_TABLE_EMPTY: return "No tensors found";
        case HMLL_ERR_TENSOR_NOT_FOUND: return "Tensor not found";
        case HMLL_ERR_CUDA_NOT_ENABLED: return "CUDA not enabled";
        case HMLL_ERR_CUDA_NO_DEVICE: return "No CUDA devices found";
        case HMLL_ERR_UNSUPPORTED_PLATFORM: return "Unsupported platform";
        case HMLL_ERR_UNSUPPORTED_FILE_FORMAT: return "Unsupported file format";
        case HMLL_ERR_UNSUPPORTED_DEVICE: return "Unsupport device";
        case HMLL_ERR_INVALID_RANGE: return "Invalid range";
        case HMLL_ERR_BUFFER_ADDR_NOT_ALIGNED: return "Pointer address not page aligned";
        case HMLL_ERR_BUFFER_TOO_SMALL: return "Buffer too small";
        case HMLL_ERR_IO_ERROR: return "Unhandled i/o error";
        case HMLL_ERR_NO_SOURCE_PROVIDED: return "No source provided";
        case HMLL_ERR_FILE_EMPTY: return "File is empty";
        case HMLL_ERR_FILE_OPEN_FAILED: return "Failed to open file";
        case HMLL_ERR_FILE_READ_FAILED: return "Failed to read file";
        case HMLL_ERR_FILE_REGISTRATION_FAILED: return "Failed to register file";
        case HMLL_ERR_MMAP_FAILED: return "Failed to memory-mapped source";
        case HMLL_ERR_IO_BUFFER_REGISTRATION_FAILED: return "Failed to register staging buffer";
        case HMLL_ERR_SAFETENSORS_JSON_INVALID_HEADER: return "Invalid safetensors json header";
        case HMLL_ERR_SAFETENSORS_JSON_MALFORMED_HEADER: return "Malformed safetensors json header";
        case HMLL_ERR_SAFETENSORS_JSON_MALFORMED_INDEX: return "Malformed safetensors json index";
        case HMLL_ERR_SYSTEM: return "OS error occured";
        case HMLL_ERR_UNKNOWN_DTYPE: return "Unknown dtype";
        default: return "Unknown error happened. Please open an issues at https://github.com/mfuntowicz/hmll/issues.";
        }
    }

    return "No error";
}

unsigned char hmll_error_is_os_error(const struct hmll_error err)
{
    return err.sys_err != HMLL_ERR_SUCCESS;
}

unsigned char hmll_error_is_lib_error(const struct hmll_error err)
{
    return err.code != HMLL_ERR_SUCCESS && err.code != HMLL_ERR_SYSTEM;
}
