#pragma once

#include <vector>
#include <set>
#include <numeric>
#include <limits>

template <typename V, typename W>
struct Edge {
  V from;
  V to;
  W wgt;
  Edge(V f, V t, W w) : from(f), to(t), wgt(w) {}
};

template <typename T, typename V, typename W>
struct Csr {
  std::vector<T> adj_ptr;
  std::vector<V> adj_node;
  std::vector<W> adj_wgt;
  std::vector<W> deg;

  void reset(int B) {
    adj_ptr.clear();
    adj_node.clear();
    adj_wgt.clear();
    deg.clear();
    deg.resize(B);
  }
};

template <typename T, typename V, typename W>
struct Graph {
  std::vector<V> partitions;
  std::vector<V> truePartitions;
  int N;
  int E;
  std::vector< Edge<V, W> > edges;
  std::vector<std::vector<std::pair<V, W>>> out_neighbors;
  std::vector<std::vector<std::pair<V, W>>> in_neighbors;
  Csr<T, V, W> csr_out;
  Csr<T, V, W> csr_in;
  std::vector<W> node_deg_out;
  std::vector<W> node_deg_in;
};

template <typename T, typename V, typename W>
struct Partition {
  int B;
  int B_to_merge;
  float S;
  int csr_out_node_size;
  int csr_in_node_size;
};

template <typename T, typename V, typename W>
struct OldData {
  Partition<T, V, W> large;
  Partition<T, V, W> med;
  Partition<T, V, W> small;

  OldData() {
    large.B = 0;
    med.B   = 0;
    small.B = 0;
    large.S = std::numeric_limits<float>::infinity();
    med.S   = std::numeric_limits<float>::infinity();
    small.S = std::numeric_limits<float>::infinity();
  }
};

template <typename V, typename W>
struct MergeData {
  std::vector<V> bestMerges;
  //V* bestMerges;
  std::vector<V> remaining_blocks;
  std::set<V>  seen;
  //std::vector<float>  dS_for_each_block;
  V* best_merge_for_each_block;
  std::vector<V> block_map;
  //std::vector<V> block_partitions;
  std::vector<V> reindex;
  void reset () {
    bestMerges.clear();
    remaining_blocks.clear();
    seen.clear();
    block_map.clear();
    reindex.clear();
    //dS_for_each_block.clear();
    //dS_for_each_block.resize(B, std::numeric_limits<float>::infinity());
    //block_partitions.clear();
    //block_partitions.resize(B);
    //std::iota(block_partitions.begin(), block_partitions.end(), 0);
  }
};

template <typename T, typename V, typename W>
struct Gpu {
 T *csr_out_adj_ptr;
 V *csr_out_adj_node;
 W *csr_out_adj_wgt;
 W *csr_out_deg;
 T *csr_in_adj_ptr;
 V *csr_in_adj_node;
 W *csr_in_adj_wgt;
 W *csr_in_deg;


 V *proposed_blocks;
 V *best_proposed_blocks;
 V *block_map_id;
 float *dS_out;
 float *dS_in;
 float *dS_new_out;
 float *dS_new_in;
 float *dS;
 float *dS_flat;


 V *proposed_blocks_nodal;

 float *dS_out_nodal;
 float *dS_in_nodal_r;
 float *dS_in_nodal_s;
 float *dS_new_r_out_nodal;
 float *dS_new_s_out_nodal;
 float *dS_new_r_in_nodal;
 float *dS_new_s_in_nodal;
 float *dS_nodal;

 W *node_deg_out;
 W *node_deg_in;

 V *partitions;

 V *random_blocks;
 float *uniform_x1;
 float *uniform_x2;
 float *uniform_x3;
 float *accepted_prob;
 V *sampling_neighbor_u;
 V *sampling_neighbor_s;

 T* g_csr_out_adj_ptr;
 V* g_csr_out_adj_node;
 V* g_csr_out_adj_block;
 W* g_csr_out_adj_wgt;
 T* g_csr_in_adj_ptr;
 V* g_csr_in_adj_node;
 V* g_csr_in_adj_block;
 W* g_csr_in_adj_wgt;

 W *g_csr_in_r_wgt;
 W *g_csr_in_s_wgt;

 W *adj_node_deg_out;
 W *adj_node_deg_in;
 float *data_S_array;
 float *dataS;

 float *itr_delta_entropy;


 W* proposed_blocks_wgt;

 // gen new csr 
  V* node_indices;
  V* block_ids;
  W* node_deg_map_out;
  W* node_deg_map_in;
  T* node_out_neighbors_ptr;
  T* node_out_neighbors_ptr_end;
  T* node_in_neighbors_ptr;
  T* node_in_neighbors_ptr_end;
  W* d_seg_out_W;//
  W* d_seg_in_W;//
  T* d_seg_out;
  T* d_seg_in;
  T* seg_flags_d_out;
  T* seg_flags_d_in;
  T* flags_d_out;
  T* flags_d_in;
  W* interm_d_out;
  W* interm_d_in;
  W* new_out_adj_wgt;
  W* new_in_adj_wgt;

  //old
  V* large_partitions;
  V* med_partitions;
  V* small_partitions;
  T* large_csr_out_ptr;
  V* large_csr_out_node;
  W* large_csr_out_wgt;
  W* large_csr_out_deg;
  T* med_csr_out_ptr;
  V* med_csr_out_node;
  W* med_csr_out_wgt;
  W* med_csr_out_deg;
  T* small_csr_out_ptr;
  V* small_csr_out_node;
  W* small_csr_out_wgt;
  W* small_csr_out_deg;

  T* large_csr_in_ptr;
  V* large_csr_in_node;
  W* large_csr_in_wgt;
  W* large_csr_in_deg;
  T* med_csr_in_ptr;
  V* med_csr_in_node;
  W* med_csr_in_wgt;
  W* med_csr_in_deg;
  T* small_csr_in_ptr;
  V* small_csr_in_node;
  W* small_csr_in_wgt;
  W* small_csr_in_deg;

  V* block_map;

};
