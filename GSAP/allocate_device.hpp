#include <cuda.h>
#include "graph.hpp"

template <typename T, typename V, typename W>
void allocate_device_memory(
  Gpu<T, V, W>& gpu,
  const Graph<T, V, W>& g,
  const Partition<T, V, W>& P,
  int BS,
  cudaStream_t* s,
  int num_proposals
) {

  int N = g.N;
  int E = g.E;
  int B = P.B;
  int tB = B*num_proposals;

  cudaMallocAsync(&gpu.node_deg_out, sizeof(W)*(4*N+6*E+2*B), s[0]);
  gpu.node_deg_in = gpu.node_deg_out + N;
  gpu.g_csr_in_r_wgt = gpu.node_deg_in + N;
  gpu.g_csr_in_s_wgt = gpu.g_csr_in_r_wgt + N;
  gpu.g_csr_out_adj_wgt = gpu.g_csr_in_s_wgt + N;
  gpu.g_csr_in_adj_wgt = gpu.g_csr_out_adj_wgt + E;
  gpu.csr_out_adj_wgt = gpu.g_csr_in_adj_wgt + E;
  gpu.csr_in_adj_wgt = gpu.csr_out_adj_wgt + E;
  gpu.adj_node_deg_out = gpu.csr_in_adj_wgt + E;
  gpu.adj_node_deg_in = gpu.adj_node_deg_out + E;
  gpu.csr_out_deg = gpu.adj_node_deg_in + E;
  gpu.csr_in_deg = gpu.csr_out_deg + B;
  


  cudaMallocAsync(&gpu.partitions, sizeof(V)*(2*N+6*E+2*B+4*tB), s[1]);
  gpu.proposed_blocks_nodal = gpu.partitions + N;
  gpu.g_csr_out_adj_node = gpu.proposed_blocks_nodal + N;
  gpu.g_csr_out_adj_block = gpu.g_csr_out_adj_node + E;
  gpu.g_csr_in_adj_node = gpu.g_csr_out_adj_block + E;
  gpu.g_csr_in_adj_block = gpu.g_csr_in_adj_node + E;
  gpu.csr_out_adj_node = gpu.g_csr_in_adj_block + E;
  gpu.csr_in_adj_node = gpu.csr_out_adj_node + E;
  gpu.best_proposed_blocks = gpu.csr_in_adj_node + E;
  gpu.block_map_id = gpu.best_proposed_blocks + B;
  gpu.proposed_blocks = gpu.block_map_id + B;
  gpu.random_blocks = gpu.proposed_blocks + tB;
  gpu.sampling_neighbor_u = gpu.random_blocks + tB;
  gpu.sampling_neighbor_s = gpu.sampling_neighbor_u + tB;


  cudaMallocAsync(&gpu.g_csr_out_adj_ptr, sizeof(T)*(2*N+2*B), s[1]);
  gpu.g_csr_in_adj_ptr = gpu.g_csr_out_adj_ptr + N;
  gpu.csr_out_adj_ptr = gpu.g_csr_in_adj_ptr + N;
  gpu.csr_in_adj_ptr = gpu.csr_out_adj_ptr + B;


  cudaMallocAsync(&gpu.dS_out, sizeof(float)*(4*B+7*tB+7*BS+E+2), s[2]);
  gpu.dS = gpu.dS_out + B;
  gpu.accepted_prob = gpu.dS + B;
  gpu.dS_out_nodal = gpu.accepted_prob + B;
  gpu.dS_in = gpu.dS_out_nodal + B;
  gpu.dS_new_out = gpu.dS_in + tB;
  gpu.dS_new_in = gpu.dS_new_out + tB;
  gpu.dS_flat = gpu.dS_new_in + tB;
  gpu.uniform_x1 = gpu.dS_flat + tB;
  gpu.uniform_x2 = gpu.uniform_x1 + tB;
  gpu.uniform_x3 = gpu.uniform_x2 + tB;
  gpu.dS_in_nodal_r = gpu.uniform_x3 + tB;
  gpu.dS_in_nodal_s = gpu.dS_in_nodal_r + BS;
  gpu.dS_new_r_out_nodal = gpu.dS_in_nodal_s + BS;
  gpu.dS_new_s_out_nodal = gpu.dS_new_r_out_nodal + BS;
  gpu.dS_new_r_in_nodal = gpu.dS_new_s_out_nodal + BS;
  gpu.dS_new_s_in_nodal = gpu.dS_new_r_in_nodal + BS;
  gpu.dS_nodal = gpu.dS_new_s_in_nodal + BS;
  gpu.data_S_array = gpu.dS_nodal + BS;
  gpu.dataS = gpu.data_S_array + E;
  gpu.itr_delta_entropy = gpu.dataS + 1;

  
  //cudaMalloc(&gpu.large_csr_out_ptr, sizeof(T)*N);
  //cudaMalloc(&gpu.med_csr_out_ptr, sizeof(T)*N);
  //cudaMalloc(&gpu.small_csr_out_ptr, sizeof(T)*N);
  //cudaMalloc(&gpu.large_csr_in_ptr, sizeof(T)*N);
  //cudaMalloc(&gpu.med_csr_in_ptr, sizeof(T)*N);
  //cudaMalloc(&gpu.small_csr_in_ptr, sizeof(T)*N);
  cudaMallocAsync(&gpu.large_csr_out_ptr, sizeof(T)*6*N, s[3]);
  gpu.med_csr_out_ptr   = gpu.large_csr_out_ptr + N; 
  gpu.small_csr_out_ptr = gpu.med_csr_out_ptr + N;
  gpu.large_csr_in_ptr  = gpu.small_csr_out_ptr + N;
  gpu.med_csr_in_ptr    = gpu.large_csr_in_ptr + N;
  gpu.small_csr_in_ptr  = gpu.med_csr_in_ptr + N;
  
  //cudaMalloc(&gpu.large_csr_out_node, sizeof(V)*E);
  //cudaMalloc(&gpu.med_csr_out_node, sizeof(V)*E);
  //cudaMalloc(&gpu.small_csr_out_node, sizeof(V)*E);
  //cudaMalloc(&gpu.large_csr_in_node, sizeof(V)*E);
  //cudaMalloc(&gpu.med_csr_in_node, sizeof(V)*E);
  //cudaMalloc(&gpu.small_csr_in_node, sizeof(V)*E);
  cudaMallocAsync(&gpu.large_csr_out_node, sizeof(V)*6*E, s[4]);
  gpu.med_csr_out_node   = gpu.large_csr_out_node + E;
  gpu.small_csr_out_node = gpu.med_csr_out_node + E;
  gpu.large_csr_in_node  = gpu.small_csr_out_node + E;
  gpu.med_csr_in_node    = gpu.large_csr_in_node + E;
  gpu.small_csr_in_node  = gpu.med_csr_in_node + E;

  //cudaMalloc(&gpu.large_csr_out_wgt, sizeof(W)*E);
  //cudaMalloc(&gpu.med_csr_out_wgt, sizeof(W)*E);
  //cudaMalloc(&gpu.small_csr_out_wgt, sizeof(W)*E);
  //cudaMalloc(&gpu.large_csr_in_wgt, sizeof(W)*E);
  //cudaMalloc(&gpu.med_csr_in_wgt, sizeof(W)*E);
  //cudaMalloc(&gpu.small_csr_in_wgt, sizeof(W)*E);
  cudaMallocAsync(&gpu.large_csr_out_wgt, sizeof(W)*6*E, s[5]);
  gpu.med_csr_out_wgt   = gpu.large_csr_out_wgt + E;
  gpu.small_csr_out_wgt = gpu.med_csr_out_wgt + E;
  gpu.large_csr_in_wgt  = gpu.small_csr_out_wgt + E;
  gpu.med_csr_in_wgt    = gpu.large_csr_in_wgt + E;
  gpu.small_csr_in_wgt  = gpu.med_csr_in_wgt + E;

  cudaMalloc(&gpu.large_csr_out_deg, sizeof(W)*N);
  cudaMalloc(&gpu.med_csr_out_deg, sizeof(W)*N);
  cudaMalloc(&gpu.small_csr_out_deg, sizeof(W)*N);
  cudaMalloc(&gpu.large_csr_in_deg, sizeof(W)*N);
  cudaMalloc(&gpu.med_csr_in_deg, sizeof(W)*N);
  cudaMalloc(&gpu.small_csr_in_deg, sizeof(W)*N);
  //cudaMallocAsync(&gpu.large_csr_out_deg, sizeof(W)*6*N, s[6]);
  //gpu.med_csr_out_deg   = gpu.large_csr_out_deg + N; 
  //gpu.small_csr_out_deg = gpu.med_csr_out_deg + N;
  //gpu.large_csr_in_deg  = gpu.large_csr_in_deg + N;
  //gpu.med_csr_in_deg    = gpu.med_csr_in_deg + N;
  //gpu.small_csr_in_deg  = gpu.small_csr_in_deg + N;

  cudaMalloc(&gpu.large_partitions, sizeof(V)*N);
  cudaMalloc(&gpu.med_partitions, sizeof(V)*N);
  cudaMalloc(&gpu.small_partitions, sizeof(V)*N);
  cudaMalloc(&gpu.node_indices, sizeof(V)*N);
  cudaMalloc(&gpu.block_ids, sizeof(V)*N);
  cudaMalloc(&gpu.block_map, sizeof(V)*N);
  //cudaMallocAsync(&gpu.large_partitions, sizeof(V)*6*N, s[7]);
  //gpu.med_partitions   = gpu.large_partitions + N;
  //gpu.small_partitions = gpu.med_partitions + N;
  //gpu.node_indices     = gpu.small_partitions + N;
  //gpu.block_ids        = gpu.node_indices + N;
  //gpu.block_map        = gpu.block_ids + N;

  cudaMalloc(&gpu.node_deg_map_out, sizeof(W)*N);
  cudaMalloc(&gpu.node_deg_map_in, sizeof(W)*N);
  cudaMalloc(&gpu.d_seg_out_W     ,sizeof(W)*N);
  cudaMalloc(&gpu.d_seg_in_W      ,sizeof(W)*N);
  //cudaMallocAsync(&gpu.node_deg_map_out, sizeof(W)*4*N, s[8]);
  //gpu.node_deg_map_in = gpu.node_deg_map_out + N;
  //gpu.d_seg_out_W     = gpu.node_deg_map_in + N;
  //gpu.d_seg_in_W      = gpu.d_seg_out_W + N;

  cudaMalloc(&gpu.new_out_adj_wgt ,sizeof(W)*E);
  cudaMalloc(&gpu.new_in_adj_wgt  ,sizeof(W)*E);
  cudaMalloc(&gpu.interm_d_out    ,sizeof(W)*(E+1));
  cudaMalloc(&gpu.interm_d_in    ,sizeof(W)*(E+1));
  //cudaMallocAsync(&gpu.new_out_adj_wgt, sizeof(W)*(4*E+2), s[9]);
  //gpu.new_in_adj_wgt = gpu.new_out_adj_wgt + E;
  //gpu.interm_d_out   = gpu.new_in_adj_wgt + E;
  //gpu.interm_d_in    = gpu.interm_d_out + E + 1;
  
  cudaMalloc(&gpu.node_out_neighbors_ptr,     sizeof(T)*N);
  cudaMalloc(&gpu.node_out_neighbors_ptr_end, sizeof(T)*N);
  cudaMalloc(&gpu.node_in_neighbors_ptr,      sizeof(T)*N);
  cudaMalloc(&gpu.node_in_neighbors_ptr_end,  sizeof(T)*N);
  cudaMalloc(&gpu.d_seg_out,       sizeof(T)*N);  
  cudaMalloc(&gpu.d_seg_in,        sizeof(T)*N);
  //cudaMallocAsync(&gpu.node_out_neighbors_ptr, sizeof(T)*6*N, s[10]);
  //gpu.node_out_neighbors_ptr_end = gpu.node_out_neighbors_ptr + N;
  //gpu.node_in_neighbors_ptr      = gpu.node_out_neighbors_ptr_end + N;
  //gpu.node_in_neighbors_ptr_end  = gpu.node_in_neighbors_ptr + N;
  //gpu.d_seg_out                  = gpu.node_in_neighbors_ptr_end  + N;
  //gpu.d_seg_in                   = gpu.d_seg_out + N;

  cudaMalloc(&gpu.seg_flags_d_out, sizeof(T)*E);
  cudaMalloc(&gpu.seg_flags_d_in,  sizeof(T)*E);
  cudaMalloc(&gpu.flags_d_out,     sizeof(T)*E);
  cudaMalloc(&gpu.flags_d_in,      sizeof(T)*E);
  //cudaMallocAsync(&gpu.seg_flags_d_out, sizeof(T)*E*4, s[11]);
  //gpu.seg_flags_d_in = gpu.seg_flags_d_out + E; 
  //gpu.flags_d_out    = gpu.seg_flags_d_in + E;
  //gpu.flags_d_in     = gpu.flags_d_out + E;

  cudaMemcpyAsync(gpu.node_deg_out, g.node_deg_out.data(), sizeof(W)*N, cudaMemcpyDefault, s[0]);
  cudaMemcpyAsync(gpu.node_deg_in, g.node_deg_in.data(), sizeof(W)*N, cudaMemcpyDefault, s[1]);
  cudaMemcpyAsync(gpu.g_csr_out_adj_ptr, g.csr_out.adj_ptr.data(), sizeof(T)*N, cudaMemcpyDefault, s[2]);
  cudaMemcpyAsync(gpu.g_csr_out_adj_node, g.csr_out.adj_node.data(), sizeof(V)*E, cudaMemcpyDefault, s[3]);
  cudaMemcpyAsync(gpu.g_csr_out_adj_wgt, g.csr_out.adj_wgt.data(), sizeof(W)*E, cudaMemcpyDefault, s[4]);
  cudaMemcpyAsync(gpu.g_csr_in_adj_ptr, g.csr_in.adj_ptr.data(), sizeof(T)*N, cudaMemcpyDefault, s[5]);
  cudaMemcpyAsync(gpu.g_csr_in_adj_node, g.csr_in.adj_node.data(), sizeof(V)*E, cudaMemcpyDefault, s[6]);
  cudaMemcpyAsync(gpu.g_csr_in_adj_wgt, g.csr_in.adj_wgt.data(), sizeof(W)*E, cudaMemcpyDefault, s[7]);


}

template <typename T, typename V, typename W>
void deallocate_device_memory(
  Gpu<T, V, W>& gpu,
  cudaStream_t& s) {

  cudaFreeAsync(gpu.node_deg_out, s);
  cudaFreeAsync(gpu.partitions, s);
  cudaFreeAsync(gpu.g_csr_out_adj_ptr, s);
  cudaFreeAsync(gpu.dS_out, s);

  //cudaFree(gpu.large_csr_out_ptr);
  //cudaFree(gpu.med_csr_out_ptr);
  //cudaFree(gpu.small_csr_out_ptr);
  //cudaFree(gpu.large_csr_in_ptr);
  //cudaFree(gpu.med_csr_in_ptr);
  //cudaFree(gpu.small_csr_in_ptr);
  cudaFreeAsync(gpu.large_csr_out_ptr, s);
  
  //cudaFree(gpu.large_csr_out_node);
  //cudaFree(gpu.med_csr_out_node);
  //cudaFree(gpu.small_csr_out_node);
  //cudaFree(gpu.large_csr_in_node);
  //cudaFree(gpu.med_csr_in_node);
  //cudaFree(gpu.small_csr_in_node);
  cudaFreeAsync(gpu.large_csr_out_node, s);

  //cudaFree(gpu.large_csr_out_wgt);
  //cudaFree(gpu.med_csr_out_wgt);
  //cudaFree(gpu.small_csr_out_wgt);
  //cudaFree(gpu.large_csr_in_wgt);
  //cudaFree(gpu.med_csr_in_wgt);
  //cudaFree(gpu.small_csr_in_wgt);
  cudaFreeAsync(gpu.large_csr_out_wgt, s);

  cudaFree(gpu.large_csr_out_deg);
  cudaFree(gpu.med_csr_out_deg);   
  cudaFree(gpu.small_csr_out_deg);
  cudaFree(gpu.large_csr_in_deg);
  cudaFree(gpu.med_csr_in_deg);
  cudaFree(gpu.small_csr_in_deg);
  //cudaFreeAsync(gpu.large_csr_out_deg, s);
  
  cudaFree(gpu.large_partitions);
  cudaFree(gpu.med_partitions);
  cudaFree(gpu.small_partitions);
  cudaFree(gpu.node_indices);
  cudaFree(gpu.block_ids);
  cudaFree(gpu.block_map);
  //cudaFreeAsync(gpu.large_partitions, s);

  cudaFree(gpu.node_deg_map_out);
  cudaFree(gpu.node_deg_map_in);
  cudaFree(gpu.d_seg_out_W    );
  cudaFree(gpu.d_seg_in_W     );
  //cudaFreeAsync(gpu.node_deg_map_out, s);

  cudaFree(gpu.new_out_adj_wgt);
  cudaFree(gpu.new_in_adj_wgt );
  cudaFree(gpu.interm_d_out   );
  cudaFree(gpu.interm_d_in   );
  //cudaFreeAsync(gpu.new_out_adj_wgt, s);

  cudaFree(gpu.node_out_neighbors_ptr);
  cudaFree(gpu.node_out_neighbors_ptr_end);
  cudaFree(gpu.node_in_neighbors_ptr);
  cudaFree(gpu.node_in_neighbors_ptr_end);
  cudaFree(gpu.d_seg_out        );
  cudaFree(gpu.d_seg_in         );
  //cudaFreeAsync(gpu.node_out_neighbors_ptr, s);

  cudaFree(gpu.seg_flags_d_out  );
  cudaFree(gpu.seg_flags_d_in   );
  cudaFree(gpu.flags_d_out      );
  cudaFree(gpu.flags_d_in       );
  //cudaFreeAsync(gpu.seg_flags_d_out, s);

}
