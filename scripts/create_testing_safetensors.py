from os import getcwd
from pathlib import Path

import torch
from safetensors.torch import save_file

DTYPES = [
    torch.complex64,
    # torch.complex128,
    # torch.float4_e2m1fn_x2,
    torch.float8_e8m0fnu,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.bfloat16,
    torch.float16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
]


if __name__ == '__main__':
    ts = {}

    for dtype in DTYPES:
        dtype_name = str(dtype)[6:]  # Remove 'torch.' prefix

        for ndim in range(6):  # 0 to 5 dimensions
            tensor_name = f"{dtype_name}.dim{ndim}"

            if ndim == 0:
                # Scalar tensor
                if dtype == torch.complex64:
                    ts[tensor_name] = torch.tensor(0 + 0j, dtype=dtype)
                else:
                    ts[tensor_name] = torch.tensor(0, dtype=dtype)
            else:
                # Create shape where each dimension has ndim items
                shape = [ndim] * ndim
                total_elements = ndim ** ndim

                if dtype == torch.complex64:
                    # For complex, create real and imaginary parts with value ndim
                    real_part = torch.full((total_elements,), ndim, dtype=torch.float32)
                    imag_part = torch.full((total_elements,), ndim, dtype=torch.float32)
                    tensor = torch.complex(real_part, imag_part).reshape(shape)
                else:
                    # Fill with value ndim
                    tensor = torch.full(shape, ndim, dtype=dtype)

                ts[tensor_name] = tensor

    cwd = getcwd()
    fpath = Path(cwd) / "hmll.safetensors"
    save_file(ts, fpath)

    print(fpath)