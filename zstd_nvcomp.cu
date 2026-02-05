/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nvcomp/zstd.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <stdexcept>

#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      std::cout << "API call failure \"" #func "\" with " << rt << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

template <typename U, typename T>
U roundUpTo(const U num, const T unit)
{
  return ((num + unit - 1) / unit) * unit;
}

int main() {
  size_t free_vram, total_vram;
  CUDA_CHECK(cudaMemGetInfo(&free_vram, &total_vram));
  // use 90% of free vram to compress
  const size_t target_vram = static_cast<size_t>(free_vram * 0.85);

  // Each batch needs ~3x memory: input + output + temp
  // chunk_size * chunk_count * 3 = target_vram
  const size_t chunk_size = 256 * 1024 * 1024; // 256MB per chunk
  const size_t memory_per_chunk = chunk_size * 3; // input + output + temp
  const size_t MAX_CHUNKS_PER_BATCH = std::max(size_t(1), target_vram / memory_per_chunk);
  const size_t BATCH_SIZE = chunk_size * MAX_CHUNKS_PER_BATCH;

  auto nvcompBatchedZstdOpts = nvcompBatchedZstdCompressDefaultOpts;

  // Query compression alignment requirements
  nvcompAlignmentRequirements_t compression_alignment_reqs;
  nvcompStatus_t status = nvcompBatchedZstdCompressGetRequiredAlignments(
      nvcompBatchedZstdOpts,
      &compression_alignment_reqs);
  if (status != nvcompSuccess) {
    throw std::runtime_error("ERROR: nvcompBatchedZstdCompressGetRequiredAlignments() not successful");
  }

  std::vector<char> input_data;
  input_data.reserve(BATCH_SIZE);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  bool is_done = false;
  while (!is_done) {
    input_data.clear();
    // Read all data from stdin
    char buffer[4096];
    while (input_data.size() < BATCH_SIZE &&
           (std::cin.read(buffer, sizeof(buffer)) || std::cin.gcount() > 0)) {
      input_data.insert(input_data.end(), buffer, buffer + std::cin.gcount());
    }

    if (input_data.empty()) {
      break;
    }

    is_done = std::cin.eof() || std::cin.gcount() == 0;

    size_t data_size = input_data.size();
    size_t chunk_count = std::min((data_size + chunk_size - 1) / chunk_size, MAX_CHUNKS_PER_BATCH);

    // Allocate aligned GPU memory for input
    const size_t aligned_chunk_size = roundUpTo(chunk_size, compression_alignment_reqs.input);
    void* d_input_data;
    CUDA_CHECK(cudaMalloc(&d_input_data, aligned_chunk_size * chunk_count));

    std::vector<void*> input_ptrs(chunk_count);
    std::vector<size_t> input_sizes(chunk_count);
    for (size_t i = 0; i < chunk_count; ++i) {
      input_ptrs[i] = static_cast<void*>(static_cast<uint8_t*>(d_input_data) + aligned_chunk_size * i);
      size_t remaining = data_size - i * chunk_size;
      input_sizes[i] = std::min(remaining, chunk_size);
    }

    // Copy data to GPU with proper alignment
    size_t offset = 0;
    for (size_t i = 0; i < chunk_count; ++i) {
      size_t copy_size = input_sizes[i];
      CUDA_CHECK(cudaMemcpy(
          input_ptrs[i],
          input_data.data() + offset,
          input_sizes[i],
          cudaMemcpyHostToDevice));
      offset += input_sizes[i];
    }

    void** d_input_ptrs;
    size_t* d_input_sizes;
    CUDA_CHECK(cudaMalloc(&d_input_ptrs, chunk_count * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&d_input_sizes, chunk_count * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_input_ptrs, input_ptrs.data(), chunk_count * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_sizes, input_sizes.data(), chunk_count * sizeof(size_t), cudaMemcpyHostToDevice));

    // Get temporary buffer size
    size_t comp_temp_bytes;
    status = nvcompBatchedZstdCompressGetTempSizeAsync(
        chunk_count,
        chunk_size,
        nvcompBatchedZstdOpts,
        &comp_temp_bytes,
        chunk_count * chunk_size);
    if (status != nvcompSuccess) {
      throw std::runtime_error("ERROR: nvcompBatchedZstdCompressGetTempSizeAsync() not successful");
    }

    void* d_comp_temp;
    CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

    // Get maximum output size
    size_t max_out_bytes;
    status = nvcompBatchedZstdCompressGetMaxOutputChunkSize(
        chunk_size, nvcompBatchedZstdOpts, &max_out_bytes);
    if (status != nvcompSuccess) {
      throw std::runtime_error("ERROR: nvcompBatchedZstdCompressGetMaxOutputChunkSize() not successful");
    }

    // Allocate aligned output buffers
    const size_t aligned_max_out_bytes = roundUpTo(max_out_bytes, compression_alignment_reqs.output);
    void* d_output_data;
    CUDA_CHECK(cudaMalloc(&d_output_data, aligned_max_out_bytes * chunk_count));

    std::vector<void*> output_ptrs(chunk_count);
    for (size_t i = 0; i < chunk_count; ++i) {
      output_ptrs[i] = static_cast<void*>(static_cast<uint8_t*>(d_output_data) + aligned_max_out_bytes * i);
    }

    void** d_output_ptrs;
    size_t* d_output_sizes;
    CUDA_CHECK(cudaMalloc(&d_output_ptrs, chunk_count * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&d_output_sizes, chunk_count * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_output_ptrs, output_ptrs.data(), chunk_count * sizeof(void*), cudaMemcpyHostToDevice));

    // Compress
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    status = nvcompBatchedZstdCompressAsync(
        d_input_ptrs,
        d_input_sizes,
        chunk_size,
        chunk_count,
        d_comp_temp,
        comp_temp_bytes,
        d_output_ptrs,
        d_output_sizes,
        nvcompBatchedZstdOpts,
        nullptr,
        stream);
    if (status != nvcompSuccess) {
      throw std::runtime_error("nvcompBatchedZstdCompressAsync() failed.");
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Get actual compressed sizes
    std::vector<size_t> compressed_sizes_host(chunk_count);
    CUDA_CHECK(cudaMemcpy(
        compressed_sizes_host.data(),
        d_output_sizes,
        chunk_count * sizeof(size_t),
        cudaMemcpyDeviceToHost));

    // Copy compressed data to host and write to stdout
    for (size_t i = 0; i < chunk_count; ++i) {
      std::vector<char> compressed_chunk(compressed_sizes_host[i]);
      CUDA_CHECK(cudaMemcpy(
          compressed_chunk.data(),
          output_ptrs[i],
          compressed_sizes_host[i],
          cudaMemcpyDeviceToHost));
      std::cout.write(compressed_chunk.data(), compressed_sizes_host[i]);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_input_data));
    CUDA_CHECK(cudaFree(d_input_ptrs));
    CUDA_CHECK(cudaFree(d_input_sizes));
    CUDA_CHECK(cudaFree(d_comp_temp));
    CUDA_CHECK(cudaFree(d_output_data));
    CUDA_CHECK(cudaFree(d_output_ptrs));
    CUDA_CHECK(cudaFree(d_output_sizes));
  }

  CUDA_CHECK(cudaStreamDestroy(stream));

  return 0;
}