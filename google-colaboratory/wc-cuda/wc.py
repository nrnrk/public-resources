import sys
import numpy as np
import pycuda.driver as cuda

BUFFER_SIZE = 1024 * 1024  # 1MB

# Initialize CUDA
cuda.init()
device = cuda.Device(0)
context = device.make_context()

# Load CUDA module
module = cuda.module_from_file("wc.cubin")
count_wc = module.get_function("count_wc")

def process_file(file_path):
    total_line_count = 0
    total_word_count = 0
    total_char_count = 0  # Count characters in python side (not CUDA side)

    with open(file_path, "rb") as f:
        while True:
            # Read file in chunks for memory efficiency
            chunk = f.read(BUFFER_SIZE)
            if not chunk:
                break

            chunk_bytes = np.frombuffer(chunk, dtype=np.uint8)
            n = len(chunk_bytes)
            total_char_count += n

            # Allocate GPU memory
            text_gpu = cuda.mem_alloc(chunk_bytes.nbytes)
            line_count_gpu = cuda.mem_alloc(np.int32(0).nbytes)
            word_count_gpu = cuda.mem_alloc(np.int32(0).nbytes)

            # Copy data to GPU
            cuda.memcpy_htod(text_gpu, chunk_bytes)
            cuda.memcpy_htod(line_count_gpu, np.array(0, dtype=np.int32))
            cuda.memcpy_htod(word_count_gpu, np.array(0, dtype=np.int32))

            # Execute kernel
            block_size = 256
            grid_size = (n + block_size - 1) // block_size
            count_wc(
                text_gpu, line_count_gpu, word_count_gpu, np.int32(n),
                block=(block_size, 1, 1), grid=(grid_size, 1)
            )

            # Copy results back to CPU
            line_count = np.zeros(1, dtype=np.int32)
            word_count = np.zeros(1, dtype=np.int32)

            cuda.memcpy_dtoh(line_count, line_count_gpu)
            cuda.memcpy_dtoh(word_count, word_count_gpu)

            # Accumulate results
            total_line_count += line_count[0]
            total_word_count += word_count[0]

            # Free GPU memory
            text_gpu.free()
            line_count_gpu.free()
            word_count_gpu.free()

    return total_line_count, total_word_count, total_char_count

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <file>")
        sys.exit(1)

    file_path = sys.argv[1]
    line_count, word_count, char_count = process_file(file_path)

    print(f" {line_count} {word_count} {char_count} {file_path}")

    # Clean up
    context.pop()
