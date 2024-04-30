#include <cuda.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thrust/reduce.h>
#include <thrust/set_operations.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/remove.h>
#include <thrust/binary_search.h>
#include <thrust/unique.h>

#include "src/moderngpu/src/moderngpu/kernel_reduce.hxx"
#include "src/moderngpu/src/moderngpu/kernel_segreduce.hxx"
#include "src/moderngpu/src/moderngpu/kernel_segsort.hxx"
#include "src/moderngpu/src/moderngpu/kernel_mergesort.hxx"
#include "src/moderngpu/src/moderngpu/kernel_scan.hxx"
#include "src/bb_segsort/bb_segsort.h"
#include "kernels.hpp"
#include "allocate_device.hpp"

template<typename T, typename V, typename W>
void carry_out_best_merges(
  Partition<T, V, W>& P,
  MergeData<V, W>& merge_data,
  const Graph<T, V, W>& g,
  Gpu<T, V, W>& gpu
) {

  int B = P.B;
  merge_data.reset();
  merge_data.block_map.resize(B);
  std::iota(merge_data.block_map.begin(), merge_data.block_map.end(), 0);

  int num_merge = 0;
  int counter = 0;
  while (num_merge < P.B_to_merge) {
    V mergeFrom = merge_data.bestMerges[counter];
    V mergeTo = merge_data.block_map[
      merge_data.best_merge_for_each_block[ 
      mergeFrom
      ]];

    counter++;
    if (mergeTo != mergeFrom) {
      for (int i = 0; i < B; i++) {
        if (merge_data.block_map[i] == mergeFrom)
          merge_data.block_map[i] = mergeTo;
      }
      num_merge += 1;
    }
  }

  for (int i = 0; i < B; i++) {
    merge_data.seen.insert(merge_data.block_map[i]);
  } 
  merge_data.reindex.resize(B, B);
  int index = 0;
  for (const auto& it : merge_data.seen) {
    merge_data.reindex[it] = index++;
  }
  for (auto& it : merge_data.block_map) {
    it = merge_data.reindex[it];
  }
  

  int block_size = 256;
  int num_blocks = (g.N + block_size - 1) / block_size;
  cudaMemcpy(gpu.block_map, merge_data.block_map.data(), sizeof(V)*B, cudaMemcpyHostToDevice);
  update_partitions<<<num_blocks, block_size>>>(gpu.partitions, gpu.block_map, g.N);

  P.B = B - P.B_to_merge;

} // end of carry_out_best_merges

template<typename T, typename V, typename W>
bool prepare_for_partition_next(
  Gpu<T, V, W>& gpu,
  const Graph<T, V, W>& g,
  OldData<T, V, W>& old,
  Partition<T, V, W>& P,
  float num_block_reduction_rate,
  cudaStream_t& s
) {

  bool optimal_B_found = false;
  int N = g.N;
  int B = P.B;

  if (P.S <= old.med.S) {
    if (old.med.B > B) {
      old.large = old.med;
      
      V* large_partitions_ori   = gpu.large_partitions;
      T* large_csr_out_ptr_ori  = gpu.large_csr_out_ptr;
      V* large_csr_out_node_ori = gpu.large_csr_out_node;
      W* large_csr_out_wgt_ori  = gpu.large_csr_out_wgt;
      W* large_csr_out_deg_ori  = gpu.large_csr_out_deg;
      T* large_csr_in_ptr_ori   = gpu.large_csr_in_ptr;
      V* large_csr_in_node_ori  = gpu.large_csr_in_node;
      W* large_csr_in_wgt_ori   = gpu.large_csr_in_wgt;
      W* large_csr_in_deg_ori   = gpu.large_csr_in_deg;

      gpu.large_partitions   = gpu.med_partitions;
      gpu.large_csr_out_ptr  = gpu.med_csr_out_ptr;
      gpu.large_csr_out_node = gpu.med_csr_out_node;
      gpu.large_csr_out_wgt  = gpu.med_csr_out_wgt;
      gpu.large_csr_out_deg  = gpu.med_csr_out_deg;
      gpu.large_csr_in_ptr   = gpu.med_csr_in_ptr;
      gpu.large_csr_in_node  = gpu.med_csr_in_node;
      gpu.large_csr_in_wgt   = gpu.med_csr_in_wgt;
      gpu.large_csr_in_deg   = gpu.med_csr_in_deg;
      
      gpu.med_partitions   = large_partitions_ori;
      gpu.med_csr_out_ptr  = large_csr_out_ptr_ori;
      gpu.med_csr_out_node = large_csr_out_node_ori;
      gpu.med_csr_out_wgt  = large_csr_out_wgt_ori;
      gpu.med_csr_out_deg  = large_csr_out_deg_ori;
      gpu.med_csr_in_ptr   = large_csr_in_ptr_ori;
      gpu.med_csr_in_node  = large_csr_in_node_ori;
      gpu.med_csr_in_wgt   = large_csr_in_wgt_ori;
      gpu.med_csr_in_deg   = large_csr_in_deg_ori;   
    }
    else {
      old.small = old.med;
      
      V* small_partitions_ori   = gpu.small_partitions;
      T* small_csr_out_ptr_ori  = gpu.small_csr_out_ptr;
      V* small_csr_out_node_ori = gpu.small_csr_out_node;
      W* small_csr_out_wgt_ori  = gpu.small_csr_out_wgt;
      W* small_csr_out_deg_ori  = gpu.small_csr_out_deg;
      T* small_csr_in_ptr_ori   = gpu.small_csr_in_ptr;
      V* small_csr_in_node_ori  = gpu.small_csr_in_node;
      W* small_csr_in_wgt_ori   = gpu.small_csr_in_wgt;
      W* small_csr_in_deg_ori   = gpu.small_csr_in_deg;

      gpu.small_partitions   = gpu.med_partitions;
      gpu.small_csr_out_ptr  = gpu.med_csr_out_ptr;
      gpu.small_csr_out_node = gpu.med_csr_out_node;
      gpu.small_csr_out_wgt  = gpu.med_csr_out_wgt;
      gpu.small_csr_out_deg  = gpu.med_csr_out_deg;
      gpu.small_csr_in_ptr   = gpu.med_csr_in_ptr;
      gpu.small_csr_in_node  = gpu.med_csr_in_node;
      gpu.small_csr_in_wgt   = gpu.med_csr_in_wgt;
      gpu.small_csr_in_deg   = gpu.med_csr_in_deg;
    
      gpu.med_partitions   = small_partitions_ori;
      gpu.med_csr_out_ptr  = small_csr_out_ptr_ori;
      gpu.med_csr_out_node = small_csr_out_node_ori;
      gpu.med_csr_out_wgt  = small_csr_out_wgt_ori;
      gpu.med_csr_out_deg  = small_csr_out_deg_ori;
      gpu.med_csr_in_ptr   = small_csr_in_ptr_ori;
      gpu.med_csr_in_node  = small_csr_in_node_ori;
      gpu.med_csr_in_wgt   = small_csr_in_wgt_ori;
      gpu.med_csr_in_deg   = small_csr_in_deg_ori;  
    }
    old.med = P;
    cudaMemcpyAsync(gpu.med_partitions, gpu.partitions, 
      sizeof(V)*N, cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(gpu.med_csr_out_ptr, gpu.csr_out_adj_ptr, 
      sizeof(T)*N, cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(gpu.med_csr_out_node, gpu.csr_out_adj_node, 
      sizeof(V)*P.csr_out_node_size, cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(gpu.med_csr_out_wgt, gpu.csr_out_adj_wgt, 
      sizeof(W)*P.csr_out_node_size, cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(gpu.med_csr_out_deg, gpu.csr_out_deg, 
      sizeof(W)*N, cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(gpu.med_csr_in_ptr, gpu.csr_in_adj_ptr, 
      sizeof(T)*N, cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(gpu.med_csr_in_node, gpu.csr_in_adj_node, 
      sizeof(V)*P.csr_in_node_size, cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(gpu.med_csr_in_wgt, gpu.csr_in_adj_wgt, 
      sizeof(W)*P.csr_in_node_size, cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(gpu.med_csr_in_deg, gpu.csr_in_deg, 
      sizeof(W)*N, cudaMemcpyDeviceToDevice, s);
  }
  else {
    if (old.med.B > P.B) {
      old.small = P;
      cudaMemcpyAsync(gpu.small_partitions, gpu.partitions, 
        sizeof(V)*N, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(gpu.small_csr_out_ptr, gpu.csr_out_adj_ptr, 
        sizeof(T)*N, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(gpu.small_csr_out_node, gpu.csr_out_adj_node, 
        sizeof(V)*P.csr_out_node_size, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(gpu.small_csr_out_wgt, gpu.csr_out_adj_wgt, 
        sizeof(W)*P.csr_out_node_size, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(gpu.small_csr_out_deg, gpu.csr_out_deg, 
        sizeof(W)*N, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(gpu.small_csr_in_ptr, gpu.csr_in_adj_ptr, 
        sizeof(T)*N, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(gpu.small_csr_in_node, gpu.csr_in_adj_node, 
        sizeof(V)*P.csr_in_node_size, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(gpu.small_csr_in_wgt, gpu.csr_in_adj_wgt, 
        sizeof(W)*P.csr_in_node_size, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(gpu.small_csr_in_deg, gpu.csr_in_deg, 
        sizeof(W)*N, cudaMemcpyDeviceToDevice, s);
    }
    else {
      old.large = P;
      cudaMemcpyAsync(gpu.large_partitions, gpu.partitions, 
        sizeof(V)*N, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(gpu.large_csr_out_ptr, gpu.csr_out_adj_ptr, 
        sizeof(T)*N, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(gpu.large_csr_out_node, gpu.csr_out_adj_node, 
        sizeof(V)*P.csr_out_node_size, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(gpu.large_csr_out_wgt, gpu.csr_out_adj_wgt, 
        sizeof(W)*P.csr_out_node_size, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(gpu.large_csr_out_deg, gpu.csr_out_deg, 
        sizeof(W)*N, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(gpu.large_csr_in_ptr, gpu.csr_in_adj_ptr, 
        sizeof(T)*N, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(gpu.large_csr_in_node, gpu.csr_in_adj_node, 
        sizeof(V)*P.csr_in_node_size, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(gpu.large_csr_in_wgt, gpu.csr_in_adj_wgt, 
        sizeof(W)*P.csr_in_node_size, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(gpu.large_csr_in_deg, gpu.csr_in_deg, 
        sizeof(W)*N, cudaMemcpyDeviceToDevice, s); 
    }
  }

  if (std::isinf(old.small.S)) {
    P = old.med;
    cudaMemcpyAsync(gpu.partitions, gpu.med_partitions, 
      sizeof(V)*N, cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(gpu.csr_out_adj_ptr, gpu.med_csr_out_ptr, 
      sizeof(T)*N, cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(gpu.csr_out_adj_node, gpu.med_csr_out_node, 
      sizeof(V)*old.med.csr_out_node_size, cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(gpu.csr_out_adj_wgt, gpu.med_csr_out_wgt, 
      sizeof(W)*old.med.csr_out_node_size, cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(gpu.csr_out_deg, gpu.med_csr_out_deg, 
      sizeof(W)*N, cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(gpu.csr_in_adj_ptr, gpu.med_csr_in_ptr, 
      sizeof(T)*N, cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(gpu.csr_in_adj_node, gpu.med_csr_in_node, 
      sizeof(V)*old.med.csr_in_node_size, cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(gpu.csr_in_adj_wgt, gpu.med_csr_in_wgt, 
      sizeof(W)*old.med.csr_in_node_size, cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(gpu.csr_in_deg, gpu.med_csr_in_deg, 
      sizeof(W)*N, cudaMemcpyDeviceToDevice, s);
    P.B_to_merge = (int)P.B * num_block_reduction_rate;
    if (P.B_to_merge == 0)  optimal_B_found = true;
  }
  else {
    if (old.large.B - old.small.B == 2) {
      optimal_B_found =   true;
      P = old.med;
      cudaMemcpyAsync(gpu.partitions, gpu.med_partitions, 
        sizeof(V)*N, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(gpu.csr_out_adj_ptr, gpu.med_csr_out_ptr, 
        sizeof(T)*N, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(gpu.csr_out_adj_node, gpu.med_csr_out_node, 
        sizeof(V)*old.med.csr_out_node_size, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(gpu.csr_out_adj_wgt, gpu.med_csr_out_wgt, 
        sizeof(W)*old.med.csr_out_node_size, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(gpu.csr_out_deg, gpu.med_csr_out_deg, 
        sizeof(W)*N, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(gpu.csr_in_adj_ptr, gpu.med_csr_in_ptr, 
        sizeof(T)*N, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(gpu.csr_in_adj_node, gpu.med_csr_in_node, 
        sizeof(V)*old.med.csr_in_node_size, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(gpu.csr_in_adj_wgt, gpu.med_csr_in_wgt, 
        sizeof(W)*old.med.csr_in_node_size, cudaMemcpyDeviceToDevice, s);
      cudaMemcpyAsync(gpu.csr_in_deg, gpu.med_csr_in_deg, 
        sizeof(W)*N, cudaMemcpyDeviceToDevice, s);
    }
    else {
      if ((old.large.B - old.med.B) >= (old.med.B - old.small.B)) {
        int next_B  = old.med.B + (int)std::round((old.large.B - old.med.B) * 0.618);
        P = old.large;
        cudaMemcpyAsync(gpu.partitions, gpu.large_partitions, 
          sizeof(V)*N, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(gpu.csr_out_adj_ptr, gpu.large_csr_out_ptr, 
          sizeof(T)*N, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(gpu.csr_out_adj_node, gpu.large_csr_out_node, 
          sizeof(V)*old.large.csr_out_node_size, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(gpu.csr_out_adj_wgt, gpu.large_csr_out_wgt, 
          sizeof(W)*old.large.csr_out_node_size, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(gpu.csr_out_deg, gpu.large_csr_out_deg, 
          sizeof(W)*N, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(gpu.csr_in_adj_ptr, gpu.large_csr_in_ptr, 
          sizeof(T)*N, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(gpu.csr_in_adj_node, gpu.large_csr_in_node, 
          sizeof(V)*old.large.csr_in_node_size, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(gpu.csr_in_adj_wgt, gpu.large_csr_in_wgt, 
          sizeof(W)*old.large.csr_in_node_size, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(gpu.csr_in_deg, gpu.large_csr_in_deg, 
          sizeof(W)*N, cudaMemcpyDeviceToDevice, s);
        P.B_to_merge = old.large.B - next_B;
      }
      else {
        int next_B  = old.small.B + (int)std::round((old.med.B - old.small.B) * 0.618);
        P = old.med;
        cudaMemcpyAsync(gpu.partitions, gpu.med_partitions, 
          sizeof(V)*N, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(gpu.csr_out_adj_ptr, gpu.med_csr_out_ptr, 
          sizeof(T)*N, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(gpu.csr_out_adj_node, gpu.med_csr_out_node, 
          sizeof(V)*old.med.csr_out_node_size, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(gpu.csr_out_adj_wgt, gpu.med_csr_out_wgt, 
          sizeof(W)*old.med.csr_out_node_size, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(gpu.csr_out_deg, gpu.med_csr_out_deg, 
          sizeof(W)*N, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(gpu.csr_in_adj_ptr, gpu.med_csr_in_ptr, 
          sizeof(T)*N, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(gpu.csr_in_adj_node, gpu.med_csr_in_node, 
          sizeof(V)*old.med.csr_in_node_size, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(gpu.csr_in_adj_wgt, gpu.med_csr_in_wgt, 
          sizeof(W)*old.med.csr_in_node_size, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(gpu.csr_in_deg, gpu.med_csr_in_deg, 
          sizeof(W)*N, cudaMemcpyDeviceToDevice, s);
        P.B_to_merge = old.med.B - next_B;
      }
    }
  }
  return optimal_B_found;
} // end of prepare_for_partition_on_next_num_blocks

template<typename T, typename V, typename W>
void propose_block_merge(
  Gpu<T, V, W>& gpu,
  Partition<T, V, W>& P,
  int B,
  MergeData<V, W>& merge_data,
  int numProposals,
  mgpu::standard_context_t& context,
  cudaStream_t *s
) {

  int block_size = 256;
  int num_blocks = (P.csr_out_node_size + block_size - 1) / block_size;
  
  cudaEvent_t e[11];
  for (int i = 0; i < 11; i++) {
    cudaEventCreate(&e[i]);
  }

  random_int_generator<V><<<num_blocks, block_size, 0, s[8]>>>(
    gpu.random_blocks, numProposals, B
  );
  cudaEventRecord(e[0], s[8]);

  uniform_number_generator<<<num_blocks, block_size, 0, s[9]>>>(
    gpu.uniform_x1, numProposals, B
  );
  cudaEventRecord(e[1], s[9]);

  uniform_number_generator<<<num_blocks, block_size, 0, s[10]>>>(
    gpu.uniform_x2, numProposals, B
  );
  cudaEventRecord(e[2], s[10]);

  uniform_number_generator<<<num_blocks, block_size, 0, s[11]>>>(
    gpu.uniform_x3, numProposals, B
  );
  cudaEventRecord(e[3], s[11]);

  calculate_accepted_prob<<<num_blocks, block_size, 0, s[12]>>>(
    gpu.accepted_prob, gpu.csr_out_deg, gpu.csr_in_deg, B
  );
  cudaEventRecord(e[4], s[12]);

  cudaStreamWaitEvent(s[0], e[1]);
  cudaStreamWaitEvent(s[0], e[2]);
  choose_neighbors<<<num_blocks, block_size, 0, s[0]>>>(
    gpu.uniform_x1, gpu.uniform_x2,
    gpu.csr_out_adj_ptr, gpu.csr_out_adj_node, 
    P.csr_out_node_size,
    gpu.csr_out_adj_wgt, gpu.csr_out_deg,
    gpu.csr_in_adj_ptr, gpu.csr_in_adj_node, 
    P.csr_in_node_size,
    gpu.csr_in_adj_wgt, gpu.csr_in_deg,
    gpu.sampling_neighbor_u, gpu.sampling_neighbor_s, numProposals, B
  );
  cudaEventRecord(e[5], s[0]);

  cudaStreamWaitEvent(s[1], e[3]);
  cudaStreamWaitEvent(s[1], e[4]);
  cudaStreamWaitEvent(s[1], e[5]);
  propose_blocks<<<num_blocks, block_size, 0, s[1]>>>(
    gpu.random_blocks, gpu.uniform_x3, gpu.accepted_prob,
    gpu.sampling_neighbor_u, gpu.sampling_neighbor_s, numProposals, B,
    gpu.csr_out_deg, gpu.csr_in_deg, gpu.proposed_blocks
  );
  cudaEventRecord(e[6], s[1]);

  calculate_dS_out<<<num_blocks, block_size, 0, s[2]>>>(
    gpu.dS_out, gpu.csr_out_adj_ptr, gpu.csr_out_adj_node, gpu.csr_out_adj_wgt,
    gpu.csr_out_deg, gpu.csr_in_deg, B, P.csr_out_node_size
  );
  cudaEventRecord(e[7], s[2]);

  
  cudaStreamWaitEvent(s[3], e[6]);
  calculate_dS_in<<<num_blocks, block_size, 0, s[3]>>>(
    gpu.dS_in, gpu.csr_in_adj_ptr, gpu.csr_in_adj_node, gpu.csr_in_adj_wgt,
    gpu.csr_in_deg, gpu.csr_out_deg, gpu.proposed_blocks, B,
    P.csr_in_node_size, numProposals
  );
  cudaEventRecord(e[8], s[3]);


  cudaStreamWaitEvent(s[4], e[6]);
  calculate_dS_new_out<<<num_blocks, block_size, 0, s[4]>>>(
    gpu.dS_new_out, gpu.proposed_blocks, gpu.csr_out_adj_ptr, gpu.csr_out_adj_node,
    gpu.csr_out_adj_wgt, gpu.csr_out_deg, gpu.csr_in_deg, B,
    P.csr_out_node_size,
    numProposals
  );
  cudaEventRecord(e[9], s[4]);

  cudaStreamWaitEvent(s[5], e[6]);
  calculate_dS_new_in<<<num_blocks, block_size, 0, s[5]>>>(
    gpu.dS_new_in, gpu.proposed_blocks, gpu.csr_in_adj_ptr, gpu.csr_in_adj_node,
    gpu.csr_in_adj_wgt, gpu.csr_in_deg, gpu.csr_out_deg, B,
    P.csr_in_node_size,
    numProposals
  );
  cudaEventRecord(e[10], s[5]);

  cudaStreamWaitEvent(s[6], e[7]);
  cudaStreamWaitEvent(s[6], e[8]);
  cudaStreamWaitEvent(s[6], e[9]);
  cudaStreamWaitEvent(s[6], e[10]);
  calculate_dS_overall<<<num_blocks, block_size, 0, s[6]>>>(
    gpu.dS_flat, gpu.dS_out, gpu.dS_in, gpu.dS_new_out, gpu.dS_new_in,
    gpu.csr_out_adj_ptr, gpu.csr_out_adj_node, gpu.csr_out_adj_wgt, gpu.csr_out_deg,
    gpu.csr_in_deg, gpu.proposed_blocks, gpu.best_proposed_blocks, B,
    numProposals
  );

  find_best_warp<V, W><<<(B*32 + 512 - 1)/512, 512, 0, s[6]>>>(
    gpu.dS, gpu.dS_flat, gpu.proposed_blocks, gpu.best_proposed_blocks, B, numProposals
  );

  cudaMemcpyAsync(merge_data.best_merge_for_each_block, gpu.best_proposed_blocks, sizeof(T)*B, cudaMemcpyDefault, s[6]);
  thrust::sequence(thrust::cuda::par.on(s[6]), gpu.block_map_id, gpu.block_map_id+B);
  thrust::device_ptr<float> t_dS(gpu.dS);
  thrust::device_ptr<V> t_block_id(gpu.block_map_id);
  thrust::sort_by_key(thrust::cuda::par.on(s[6]), t_dS, t_dS+B, t_block_id);
  merge_data.bestMerges.resize(B);
  cudaMemcpyAsync(&merge_data.bestMerges[0], gpu.block_map_id, sizeof(V)*B, cudaMemcpyDefault, s[6]);


  for (int i = 0; i < 11; i++) {
    cudaEventDestroy(e[i]);
  }

} // end of propose_block_merge

template <typename T, typename V, typename W>
void propose_nodal_move(
  Gpu<T, V, W>& gpu,
  Graph<T, V, W>& g,
  Partition<T, V, W>& P,
  float& itr_delta_entropy,
  int itr,
  int BS,
  int N,
  int B,
  mgpu::standard_context_t& context,
  cudaStream_t *s
) {

  int E = g.E;

  int block_size = 256;
  int num_blocks = (g.E + block_size - 1) / block_size;

  cudaEvent_t e[14];
  for (int i = 0; i < 14; i++) {
    cudaEventCreate(&e[i]);
  }


  random_int_generator_nodal<V><<<num_blocks, block_size, 0, s[8]>>>(
    gpu.random_blocks, B, BS
  );

  uniform_number_generator<<<num_blocks, block_size, 0, s[9]>>>(
    gpu.uniform_x1, 1, BS
  );

  uniform_number_generator<<<num_blocks, block_size, 0, s[10]>>>(
    gpu.uniform_x2, 1, B
  );

  uniform_number_generator<<<num_blocks, block_size, 0, s[11]>>>(
    gpu.uniform_x3, 1, B
  );
  cudaEventRecord(e[0], s[11]);
  
  calculate_accepted_prob<<<num_blocks, block_size, 0, s[12]>>>(
    gpu.accepted_prob, gpu.csr_out_deg, gpu.csr_in_deg, B
  );
  cudaEventRecord(e[1], s[12]);


  choose_neighbored_node_block<<<num_blocks, block_size, 0, s[9]>>>(
    gpu.uniform_x1,
    gpu.g_csr_out_adj_ptr, gpu.g_csr_out_adj_node, g.E, gpu.g_csr_out_adj_wgt, gpu.node_deg_out,
    gpu.g_csr_in_adj_ptr, gpu.g_csr_in_adj_node, g.E, gpu.g_csr_in_adj_wgt, gpu.node_deg_in,
    gpu.partitions, gpu.sampling_neighbor_u, itr, BS, N
  );
  cudaEventRecord(e[2], s[9]);

  
  cudaStreamWaitEvent(s[10], e[0]);
  choose_neighbored_block<<<num_blocks, block_size, 0, s[10]>>>(
    gpu.uniform_x2,
    gpu.csr_out_adj_ptr, gpu.csr_out_adj_node, 
    P.csr_out_node_size,
    gpu.csr_out_adj_wgt, gpu.csr_out_deg,
    gpu.csr_in_adj_ptr, gpu.csr_in_adj_node, 
    P.csr_in_node_size,
    gpu.csr_in_adj_wgt, gpu.csr_in_deg,
    gpu.sampling_neighbor_s, B
  );
  cudaEventRecord(e[3], s[10]);

  cudaStreamWaitEvent(s[0], e[0]);
  cudaStreamWaitEvent(s[0], e[1]);
  cudaStreamWaitEvent(s[0], e[2]);
  cudaStreamWaitEvent(s[0], e[3]);
  propose_nodal<<<num_blocks, block_size, 0, s[0]>>>(
    gpu.random_blocks, gpu.uniform_x3, gpu.accepted_prob, gpu.node_deg_out, gpu.node_deg_in,
    gpu.sampling_neighbor_u, gpu.sampling_neighbor_s, BS, itr, gpu.proposed_blocks_nodal
  );
  cudaEventRecord(e[4], s[0]);


  match_neighbored_block_out<T, V, W><<<num_blocks, block_size, 0, s[1]>>>(
    gpu.g_csr_out_adj_ptr, gpu.g_csr_out_adj_node, g.E,
    gpu.g_csr_out_adj_block, gpu.partitions, N
  );
  match_neighbored_block_in<T, V, W><<<num_blocks, block_size, 0, s[1]>>>(
    gpu.g_csr_in_adj_ptr, gpu.g_csr_in_adj_node, g.E,
    gpu.g_csr_in_adj_block, gpu.partitions, N
  );

  bb_segsort(gpu.g_csr_out_adj_block, gpu.g_csr_out_adj_wgt, g.E, gpu.g_csr_out_adj_ptr, N);
  bb_segsort(gpu.g_csr_in_adj_block, gpu.g_csr_in_adj_wgt, g.E, gpu.g_csr_in_adj_ptr, N);
  cudaEventRecord(e[5], s[1]);
  
  calculate_dS_out<<<num_blocks, block_size, 0, s[2]>>>(
    gpu.dS_out_nodal, gpu.csr_out_adj_ptr, gpu.csr_out_adj_node, gpu.csr_out_adj_wgt,
    gpu.csr_out_deg, gpu.csr_in_deg, B, P.csr_out_node_size
  );
  cudaEventRecord(e[6], s[2]);


  cudaStreamWaitEvent(s[3], e[4]);
  calculate_dS_in_nodal_r<<<num_blocks, block_size, 0, s[3]>>>(
    gpu.dS_in_nodal_r, gpu.csr_in_adj_ptr, gpu.csr_in_adj_node, gpu.csr_in_adj_wgt,
    gpu.csr_in_deg, gpu.csr_out_deg, gpu.partitions, gpu.proposed_blocks_nodal,
    BS, itr, B, N, P.csr_in_node_size
  );
  cudaEventRecord(e[7], s[3]);

  cudaStreamWaitEvent(s[4], e[4]);
  calculate_dS_in_nodal_s<<<num_blocks, block_size, 0, s[4]>>>(
    gpu.dS_in_nodal_s, gpu.csr_in_adj_ptr, gpu.csr_in_adj_node, gpu.csr_in_adj_wgt,
    gpu.csr_in_deg, gpu.csr_out_deg, gpu.partitions, gpu.proposed_blocks_nodal,
    BS, itr, B, N, P.csr_out_node_size
  );
  cudaEventRecord(e[8], s[4]);

  cudaStreamWaitEvent(s[5], e[5]);
  find_csr_in_r_wgt<<<num_blocks, block_size, 0, s[5]>>>(
    gpu.g_csr_in_r_wgt, gpu.partitions,
    gpu.g_csr_in_adj_ptr, gpu.g_csr_in_adj_block, g.csr_in.adj_node.size(), gpu.g_csr_in_adj_wgt, 
    BS, itr, B, N
  );

  calculate_dS_new_r_out_nodal_new<<<num_blocks, block_size, 0, s[5]>>>(
    gpu.dS_new_r_out_nodal, gpu.partitions, gpu.proposed_blocks_nodal,
    gpu.csr_out_adj_ptr, gpu.csr_out_adj_node, 
    P.csr_out_node_size,
    gpu.csr_out_adj_wgt,
    gpu.g_csr_out_adj_ptr, gpu.g_csr_out_adj_block, g.csr_out.adj_node.size(), gpu.g_csr_out_adj_wgt,
    gpu.node_deg_out, gpu.csr_in_deg, gpu.csr_out_deg, gpu.g_csr_in_r_wgt,
    BS, itr, B, N
  );
  cudaEventRecord(e[9], s[5]);

  cudaStreamWaitEvent(s[6], e[5]);
  find_csr_in_s_wgt<T, V, W><<<num_blocks, block_size, 0, s[6]>>>(
    gpu.g_csr_in_s_wgt, gpu.proposed_blocks_nodal,
    gpu.g_csr_in_adj_ptr, gpu.g_csr_in_adj_block, g.csr_in.adj_node.size(), gpu.g_csr_in_adj_wgt, 
    BS, itr, B, N
  );

  calculate_dS_new_s_out_nodal_new<<<num_blocks, block_size, 0, s[6]>>>(
    gpu.dS_new_s_out_nodal, gpu.partitions, gpu.proposed_blocks_nodal,
    gpu.csr_out_adj_ptr, gpu.csr_out_adj_node, 
    P.csr_out_node_size,
    gpu.csr_out_adj_wgt,
    gpu.g_csr_out_adj_ptr, gpu.g_csr_out_adj_block, g.csr_out.adj_node.size(), gpu.g_csr_out_adj_wgt,
    gpu.node_deg_out, gpu.csr_in_deg, gpu.csr_out_deg, gpu.g_csr_in_s_wgt,
    BS, itr, B, N
  );
  cudaEventRecord(e[10], s[6]);

  cudaStreamWaitEvent(s[7], e[5]);
  calculate_dS_new_r_in_nodal_new<<<num_blocks, block_size, 0, s[7]>>>(
    gpu.dS_new_r_in_nodal, gpu.partitions, gpu.proposed_blocks_nodal,
    gpu.csr_in_adj_ptr, gpu.csr_in_adj_node, 
    P.csr_in_node_size,
    gpu.csr_in_adj_wgt,
    gpu.g_csr_in_adj_ptr, gpu.g_csr_in_adj_block, g.csr_in.adj_node.size(), gpu.g_csr_in_adj_wgt,
    gpu.node_deg_in, gpu.csr_in_deg, gpu.csr_out_deg,
    BS, itr, B, N
  );
  cudaEventRecord(e[11], s[7]);

  cudaStreamWaitEvent(s[8], e[5]);
  calculate_dS_new_s_in_nodal_new<<<num_blocks, block_size, 0, s[8]>>>(
    gpu.dS_new_s_in_nodal, gpu.partitions, gpu.proposed_blocks_nodal,
    gpu.csr_in_adj_ptr, gpu.csr_in_adj_node, 
    P.csr_in_node_size,
    gpu.csr_in_adj_wgt,
    gpu.g_csr_in_adj_ptr, gpu.g_csr_in_adj_block, g.csr_in.adj_node.size(), gpu.g_csr_in_adj_wgt,
    gpu.node_deg_in, gpu.csr_in_deg, gpu.csr_out_deg,
    BS, itr, B, N
  );
  cudaEventRecord(e[12], s[8]);


  cudaStreamWaitEvent(s[9], e[6]);
  cudaStreamWaitEvent(s[9], e[7]);
  cudaStreamWaitEvent(s[9], e[8]);
  cudaStreamWaitEvent(s[9], e[9]);
  cudaStreamWaitEvent(s[9], e[10]);
  cudaStreamWaitEvent(s[9], e[11]);
  cudaStreamWaitEvent(s[9], e[12]);
  calculate_dS_overall_nodal<V, W><<<num_blocks, block_size, 0, s[9]>>>(
    gpu.dS_nodal,
    gpu.dS_new_r_in_nodal, gpu.dS_new_s_in_nodal,
    gpu.dS_new_s_out_nodal, gpu.dS_new_r_out_nodal,
    gpu.dS_in_nodal_r, gpu.dS_in_nodal_s,
    gpu.dS_out_nodal,
    gpu.partitions, gpu.proposed_blocks_nodal,
    BS, itr, B, N
  );
  cudaEventRecord(e[13], s[9]);

  float delta_entropy_itr = 0;
  cudaStreamWaitEvent(context.stream(), e[13]);
  mgpu::reduce(gpu.dS_nodal, BS, gpu.itr_delta_entropy, mgpu::plus_t<float>(), context);
  cudaMemcpyAsync(&delta_entropy_itr, gpu.itr_delta_entropy, sizeof(float), cudaMemcpyDefault, context.stream());
  itr_delta_entropy += delta_entropy_itr;

  for (int i = 0; i < 14; i++) {
    cudaEventDestroy(e[i]);
  }

} // end of propose_nodal_move


template <typename T, typename V, typename W>
float compute_overall_entropy_cuda(
  Gpu<T, V, W>& gpu,
  Graph<T, V, W>& g,
  Partition<T, V, W>& P,
  int B,
  mgpu::standard_context_t& context) {

  int block_size = 512;
  int num_blocks = (g.E + block_size - 1) / block_size;

  block_deg_mapping<<<num_blocks, block_size, 0, context.stream()>>>(
    gpu.csr_out_adj_ptr, gpu.csr_out_adj_node, P.csr_out_node_size, gpu.csr_out_adj_wgt,
    gpu.adj_node_deg_out, gpu.adj_node_deg_in, gpu.csr_out_deg, gpu.csr_in_deg, B
  );

  compute_overall_S_transform<<<num_blocks, block_size, 0, context.stream()>>>(
    gpu.csr_out_adj_ptr, gpu.csr_out_adj_node, P.csr_out_node_size, gpu.csr_out_adj_wgt,
    gpu.data_S_array, gpu.adj_node_deg_out, gpu.adj_node_deg_in, B
  );


  float data_S = 0;
  mgpu::reduce(
    gpu.data_S_array, P.csr_out_node_size,
    gpu.dataS, mgpu::plus_t<float>(), context
  );
  
  cudaMemcpyAsync(&data_S, gpu.dataS, sizeof(float), cudaMemcpyDefault, context.stream());


  float model_S_term = (float)B*B/g.E;

  float model_S = (float)(g.E * (1 + model_S_term) * std::log(1 + model_S_term)) -
                    (model_S_term * std::log(model_S_term)) + (g.N * std::log(B));

  return model_S - data_S;


} // end of compute_overall_entropy_cuda


template<typename T, typename V, typename W>
void calculate_new_csr(
  Partition<T, V, W>& P,
  Gpu<T, V, W>& gpu,
  const Graph<T, V, W>& g,
  mgpu::standard_context_t& context,
  cudaStream_t *s
) {
  
  int N = g.N;
  int E = g.E;
  int B = P.B;

  int block_size = 256;
  int num_blocks = (E + block_size - 1) / block_size;

  cudaMemset(gpu.csr_out_deg, 0, sizeof(W)*B);
  cudaMemset(gpu.csr_in_deg, 0, sizeof(W)*B);

  thrust::sequence(thrust::device, gpu.node_indices, gpu.node_indices + N);
  thrust::copy(thrust::device, gpu.partitions, gpu.partitions + N, gpu.block_ids);
  thrust::sort_by_key(thrust::device, gpu.block_ids, gpu.block_ids + N, gpu.node_indices);

  node_deg_mapping<<<num_blocks, block_size>>>(
    gpu.node_deg_map_out, gpu.node_indices, gpu.node_deg_out, N
  );
  node_deg_mapping<<<num_blocks, block_size>>>(
    gpu.node_deg_map_in, gpu.node_indices, gpu.node_deg_in, N
  );
  
  thrust::exclusive_scan(thrust::device, 
    gpu.node_deg_map_out, gpu.node_deg_map_out + N, gpu.node_out_neighbors_ptr
  );
  thrust::inclusive_scan(thrust::device, 
    gpu.node_deg_map_out, gpu.node_deg_map_out + N, gpu.node_out_neighbors_ptr_end
  );
  thrust::exclusive_scan(thrust::device, 
    gpu.node_deg_map_in, gpu.node_deg_map_in + N, gpu.node_in_neighbors_ptr
  );
  thrust::inclusive_scan(thrust::device, 
    gpu.node_deg_map_in, gpu.node_deg_map_in + N, gpu.node_in_neighbors_ptr_end
  );

  update_node_neighbors_block_ids<<<num_blocks, block_size>>>(
    gpu.csr_out_adj_node, gpu.csr_out_adj_wgt,
    gpu.node_indices, gpu.node_out_neighbors_ptr, gpu.node_out_neighbors_ptr_end,
    gpu.g_csr_out_adj_ptr, gpu.g_csr_out_adj_node, E,
    gpu.g_csr_out_adj_wgt, gpu.partitions, gpu.csr_out_deg, N
  );
  update_node_neighbors_block_ids<<<num_blocks, block_size>>>(
    gpu.csr_in_adj_node, gpu.csr_in_adj_wgt,
    gpu.node_indices, gpu.node_in_neighbors_ptr, gpu.node_in_neighbors_ptr_end,
    gpu.g_csr_in_adj_ptr, gpu.g_csr_in_adj_node, E,
    gpu.g_csr_in_adj_wgt, gpu.partitions, gpu.csr_in_deg, N
  );

  thrust::exclusive_scan(thrust::device, 
    gpu.csr_out_deg, gpu.csr_out_deg + B, gpu.d_seg_out_W
  );
  thrust::exclusive_scan(thrust::device, 
    gpu.csr_in_deg, gpu.csr_in_deg + B, gpu.d_seg_in_W
  );

  thrust::transform(thrust::device, 
    gpu.d_seg_out_W, gpu.d_seg_out_W + B, gpu.d_seg_out, W_to_T<T, W>()
  );
  thrust::transform(thrust::device, 
    gpu.d_seg_in_W, gpu.d_seg_in_W + B, gpu.d_seg_in, W_to_T<T, W>()
  );

  bb_segsort(gpu.csr_out_adj_node, gpu.csr_out_adj_wgt, E, gpu.d_seg_out, B);
  bb_segsort(gpu.csr_in_adj_node, gpu.csr_in_adj_wgt, E, gpu.d_seg_in, B);
  cudaMemset(gpu.seg_flags_d_out,  0, sizeof(T)*E);
  cudaMemset(gpu.seg_flags_d_in,   0, sizeof(T)*E);
  cudaMemset(gpu.flags_d_out,      0, sizeof(T)*E);
  cudaMemset(gpu.flags_d_in,       0, sizeof(T)*E);
  cudaMemset(gpu.interm_d_out,     0, sizeof(W)*(E+1));
  cudaMemset(gpu.interm_d_in,      0, sizeof(W)*(E+1));

  fill_seg_flags<T, V, W><<<num_blocks, block_size>>>(
    gpu.d_seg_out, gpu.seg_flags_d_out, B
  );
  fill_seg_flags<T, V, W><<<num_blocks, block_size>>>(
    gpu.d_seg_in, gpu.seg_flags_d_in, B
  );
  compute_subseg_flags<<<num_blocks, block_size>>>(
    gpu.seg_flags_d_out, gpu.csr_out_adj_node, gpu.flags_d_out, gpu.interm_d_out, E
  );
  compute_subseg_flags<<<num_blocks, block_size>>>(
    gpu.seg_flags_d_in, gpu.csr_in_adj_node, gpu.flags_d_in, gpu.interm_d_in, E
  );

  thrust::device_ptr<V> thp_flags_d_out = thrust::device_pointer_cast<V>(gpu.flags_d_out);
  thrust::device_ptr<V> thp_flags_d_in = thrust::device_pointer_cast<V>(gpu.flags_d_in);
  thrust::device_ptr<V> thp_d_key_out = thrust::device_pointer_cast<V>(gpu.csr_out_adj_node);
  thrust::device_ptr<V> thp_d_key_in = thrust::device_pointer_cast<V>(gpu.csr_in_adj_node);
  
  auto key_new_end_out  = thrust::remove_if(thrust::device,
    thp_d_key_out+1, thp_d_key_out+E, thp_flags_d_out+1, is_zero()
  );
  auto key_new_end_in   = thrust::remove_if(thrust::device,
    thp_d_key_in+1, thp_d_key_in+E, thp_flags_d_in+1, is_zero()
  );
  auto flag_new_end_out = thrust::remove(thrust::device,
    thp_flags_d_out+1, thp_flags_d_out+E, 0
  );
  auto flag_new_end_in  = thrust::remove(thrust::device,
    thp_flags_d_in+1, thp_flags_d_in+E, 0
  );

  thrust::device_ptr<W> thp_d_seg_out = thrust::device_pointer_cast<W>(gpu.interm_d_out);
  thrust::device_ptr<W> thp_d_seg_in = thrust::device_pointer_cast<W>(gpu.interm_d_in);
 
  thrust::exclusive_scan(thrust::device,
    thp_d_seg_out, thp_d_seg_out + E + 1, thp_d_seg_out
  );
  thrust::exclusive_scan(thrust::device,
    thp_d_seg_in, thp_d_seg_in + E + 1, thp_d_seg_in
  );
  
  P.csr_out_node_size = flag_new_end_out - thp_flags_d_out;
  P.csr_in_node_size = flag_new_end_in - thp_flags_d_in;
  
  mgpu::segreduce(
    gpu.csr_out_adj_wgt, E, thp_flags_d_out, P.csr_out_node_size, gpu.new_out_adj_wgt, 
    mgpu::plus_t<W>(), (W)0, context
  );
  mgpu::segreduce(
    gpu.csr_in_adj_wgt, E, thp_flags_d_in, P.csr_in_node_size, gpu.new_in_adj_wgt, 
    mgpu::plus_t<W>(), (W)0, context
  );
  
  gpu.csr_out_adj_wgt = gpu.new_out_adj_wgt;
  gpu.csr_in_adj_wgt = gpu.new_in_adj_wgt;

  map_adj_ptr<T><<<num_blocks, block_size>>>(
    gpu.csr_out_adj_ptr, gpu.d_seg_out, gpu.interm_d_out, B
  );
  map_adj_ptr<T><<<num_blocks, block_size>>>(
    gpu.csr_in_adj_ptr, gpu.d_seg_in, gpu.interm_d_in, B
  );

}


template<typename T, typename V, typename W>
void partition(Graph<T, V, W>& g) {

  int   num_agg_proposals_per_block     = 10;
  float num_block_reduction_rate        = 0.35;
  int   max_num_nodal_itr               = 100;
  int   num_batch_nodal_update          = 4;
  float delta_entropy_threshold1        = 5e-4;
  float delta_entropy_threshold2        = 1e-4;
  int   delta_entropy_moving_avg_window = 3;


  OldData<T, V, W> old;
  Partition<T, V, W> P;
  MergeData<V, W> merge_data;
  std::vector<float> itr_delta_entropy;

  cudaMallocHost(&merge_data.best_merge_for_each_block, sizeof(V)*g.N);

  P.B = g.N;
  P.csr_out_node_size = g.E;
  P.csr_in_node_size = g.E;

  P.B_to_merge = (int)P.B * num_block_reduction_rate;

  Gpu<T, V, W> gpu;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaStream_t s[15];
  for (int i = 0; i < 15; i++) {
    cudaStreamCreate(&s[i]);
  }

  allocate_device_memory<T, V, W>(gpu, g, P, g.N/num_batch_nodal_update, s, num_agg_proposals_per_block);

  thrust::sequence(thrust::device, gpu.partitions, gpu.partitions + P.B);

  mgpu::standard_context_t context(false);

  calculate_new_csr<T, V, W>(P, gpu, g, context, s);


  float S = 0;

  bool optimal_B_found = false;
  while (!optimal_B_found) {

    propose_block_merge<T, V, W>(gpu, P, P.B, merge_data, num_agg_proposals_per_block, context, s);

    carry_out_best_merges<T, V, W>(P, merge_data, g, gpu);

    calculate_new_csr<T, V, W>(P, gpu, g, context, s);

    itr_delta_entropy.clear();
    itr_delta_entropy.resize(max_num_nodal_itr, 0.0);


    // nodal updates
    int batch_size = g.N / num_batch_nodal_update;
    for (int itr = 0; itr < max_num_nodal_itr; itr++) {
      for (int b_itr = 0; b_itr < num_batch_nodal_update; b_itr++) {
        propose_nodal_move<T, V, W>(gpu, g, P, itr_delta_entropy[itr], b_itr, batch_size, g.N, P.B, context, s);
        calculate_new_csr<T, V, W>(P, gpu, g, context, s);      
      } // num_batch_nodal_update

      S = compute_overall_entropy_cuda<T, V, W>(gpu, g, P, P.B, context);

      if (itr >= (delta_entropy_moving_avg_window - 1)) {
        bool isf = std::isfinite(old.large.S) && std::isfinite(old.med.S)
          && std::isfinite(old.small.S);
        float mean = 0;
        for (int i = itr - delta_entropy_moving_avg_window + 1; i < itr; i++) {
          mean += itr_delta_entropy[i];
        }
        mean /= (float)(delta_entropy_moving_avg_window - 1);
        if (!isf) {
          if (-mean < (delta_entropy_threshold1 * S)) break;
        }
        else {
          if (-mean < (delta_entropy_threshold2 * S)) break;
        }
      }
    } // end itr
    
    P.S = S;
    optimal_B_found = prepare_for_partition_next<T, V, W>(gpu, g, old, P, num_block_reduction_rate, stream);
  } // end while
  g.partitions.resize(g.N);
  cudaMemcpy(&g.partitions[0], gpu.partitions, sizeof(V)*g.N, cudaMemcpyDeviceToHost);

  deallocate_device_memory(gpu, stream);

  cudaStreamDestroy(stream);

  for (int i = 0; i < 12; i++) {
    cudaStreamDestroy(s[i]);
  }

  cudaFreeHost(merge_data.best_merge_for_each_block);


} // end of partition

