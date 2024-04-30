#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

template<typename T, typename W>
struct W_to_T
{
  __host__ __device__
  T operator()(W x) const {
    return static_cast<T>(x);
  }
};


struct is_zero
{
  __host__ __device__
  bool operator()(const int &x) const {
    return x == 0;
  }
};

template<typename V, typename W>
__global__ void node_deg_mapping(
  W* nodes_deg_out_map,
  V* nodes_index,
  W* g_csr_out_deg,
  int N
) {

  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    nodes_deg_out_map[idx] = g_csr_out_deg[nodes_index[idx]];
  }

}

template<typename T, typename V, typename W>
__global__ void update_node_neighbors_block_ids(
  V* nodes_neighbor_block,
  W* nodes_neighbor_weight,
  V* nodes_index,
  T* nodes_neighbor_ptr,
  T* nodes_neighbor_ptr_end,
  T* g_csr_out_adj_ptr,
  V* g_csr_out_adj_node,
  size_t g_csr_out_adj_node_size,
  W* g_csr_out_adj_wgt,
  V* partitions,
  W* csr_out_deg, //
  int N
) {

  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    V node = nodes_index[idx]; // node is sorted based on blockid
    T ptr_begin = nodes_neighbor_ptr[idx];
    T ptr_end = nodes_neighbor_ptr_end[idx];
    T g_ptr_begin = g_csr_out_adj_ptr[node];
    T g_ptr_end;
    if (node + 1 < N) {
      g_ptr_end = g_csr_out_adj_ptr[node+1];
    }
    else {
      g_ptr_end = g_csr_out_adj_node_size;
    }
    W deg = 0;
    for (T i = ptr_begin, j = g_ptr_begin; i < ptr_end && j < g_ptr_end; i++, j++) {
      nodes_neighbor_block[i] = partitions[g_csr_out_adj_node[j]];
      W wgt = g_csr_out_adj_wgt[j];
      nodes_neighbor_weight[i] = wgt;
      deg += wgt;
    }
    V block = partitions[node];
    atomicAdd(&csr_out_deg[block], deg);
  }

}

template<typename T, typename V, typename W>
__global__ void fill_seg_flags(
  T* segs, 
  T* seg_flags, 
  int length
) {

  unsigned tid = threadIdx.x + blockDim.x*blockIdx.x;
  if (tid < length) {
    seg_flags[segs[tid]] = 1;
  }

}

template<typename T, typename V, typename W>
__global__ void compute_subseg_flags(
  T* seg_flags, 
  V* keys, 
  T* flags, 
  W* interm, 
  int n
) {

  unsigned tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < n) {
    if (tid == 0) {
      interm[tid] = 1;
    }
    else {
      if (keys[tid] != keys[tid-1]) {
        flags[tid] = tid;
        interm[tid] = 1;
      }
      if (seg_flags[tid]) {
        flags[tid] = tid;
        interm[tid] = 1;
      }
    }
  }

}


template<typename T, typename W>
__global__ void map_adj_ptr(
  T* csr_adj_ptr,
  T* d_seg,
  W* d_new_seg,
  int B
) {

  unsigned idx = threadIdx.x + blockDim.x*blockIdx.x;

  if (idx < B) { 
    csr_adj_ptr[idx] = d_new_seg[d_seg[idx]]; 
  }

}


template<typename T, typename V, typename W>
__global__ void calculate_dS_out(
  float* dS_out,
  T* csr_out_adj_ptr,
  V* csr_out_adj_node,
  W* csr_out_adj_wgt,
  W* csr_out_deg,
  W* csr_in_deg,
  int B,
  size_t csr_out_size
) {


  unsigned r = blockIdx.x * blockDim.x + threadIdx.x;

  if (r < B) {
    T ptr_s = csr_out_adj_ptr[r];
    T ptr_e;
    if (r + 1 < B) {
      ptr_e = csr_out_adj_ptr[r+1];
    }
    else {
      ptr_e = csr_out_size;
    }
    W deg_out_r = csr_out_deg[r];
    float dS = 0;
    float log_csr_out_deg_r = __log2f(deg_out_r);
    for (T i = ptr_s; i < ptr_e; i++) {
      W w = csr_out_adj_wgt[i];
      V node = csr_out_adj_node[i];
      float log_w = __log2f(w);
      W csr_in_deg_i = csr_in_deg[node];
      float log_csr_in_deg_i = __log2f(csr_in_deg_i);
      float p = log_w - ( log_csr_out_deg_r + log_csr_in_deg_i);
      dS += (float)w*p;
      //dS += (float)w * __log2f((float)w/(deg_out_r * csr_in_deg[csr_out_adj_node[i]]));
    }
    dS_out[r] = dS;
  }
}


template<typename T, typename V, typename W>
__global__ void calculate_dS_in(
  float* dS_in,
  T* csr_in_adj_ptr,
  V* csr_in_adj_node,
  W* csr_in_adj_wgt,
  W* csr_in_deg,
  W* csr_out_deg,
  V* proposed_blocks,
  int B,
  size_t csr_in_size,
  int num_proposals
) {

  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < B*num_proposals) {
    V r = tid / num_proposals;
    V s = proposed_blocks[tid];
    T ptr_s = csr_in_adj_ptr[r];
    T ptr_e;
    if (r + 1 < B) { 
      ptr_e = csr_in_adj_ptr[r+1];
    }    
    else {
      ptr_e = csr_in_size;
    }    
    float dS = 0; 
    W in_deg = csr_in_deg[r];
    float log_csr_in_deg_r = __log2f(in_deg);
    for (T i = ptr_s; i < ptr_e; i++) {
      V n = csr_in_adj_node[i];
      if (n != r && n != s) {
        W w = csr_in_adj_wgt[i];
        float log_w = __log2f(w);
        float log_csr_out_deg_i = __log2f(csr_out_deg[n]);
        float p = log_w - ( log_csr_out_deg_i + log_csr_in_deg_r);
        dS += (float)w*p;
      }
      //if (n != r && n != s) { 
      //  W w = csr_in_adj_wgt[i];
      //  dS += (float)w * __log2f((float)w / (csr_out_deg[n] * in_deg));
      //} 
    }
    dS_in[tid] = dS;
  }
}

template<typename T, typename V, typename W>
__global__ void calculate_dS_new_out(
  float* dS_new_out,
  V* proposed_blocks,
  T* csr_out_adj_ptr,
  V* csr_out_adj_node,
  W* csr_out_adj_wgt,
  W* csr_out_deg,
  W* csr_in_deg,
  int B,
  size_t csr_out_size,
  int num_proposals
) {

  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < B*num_proposals) {
    V r = tid / num_proposals;
    V s = proposed_blocks[tid];
    T ptr_r_s, ptr_r_e;
    T ptr_s_s, ptr_s_e;
    ptr_r_s = csr_out_adj_ptr[r];
    ptr_s_s = csr_out_adj_ptr[s];
    if (r + 1 < B) {
      ptr_r_e = csr_out_adj_ptr[r+1];
    }
    else {
      ptr_r_e = csr_out_size;
    }
    if (s + 1 < B) {
      ptr_s_e = csr_out_adj_ptr[s+1];
    }
    else {
      ptr_s_e = csr_out_size;
    }

    T i = ptr_r_s, j = ptr_s_s;
    float dS = 0;
    W w;
    W d_out_new_s = csr_out_deg[s] + csr_out_deg[r];
    float log_d_out_new_s = __log2f(d_out_new_s);
    V n, n1, n2;
    while (i < ptr_r_e && j < ptr_s_e) {
      n1 = csr_out_adj_node[i];
      n2 = csr_out_adj_node[j];
      if (n1 < n2) {
        w = csr_out_adj_wgt[i];
        n = n1;
        i++;
      }
      else if (n1 > n2) {
        w = csr_out_adj_wgt[j];
        n = n2;
        j++;
      }
      else {
        w = csr_out_adj_wgt[i] + csr_out_adj_wgt[j];
        n = n1;
        i++;
        j++;
      }
      //dS -= w * __log2f((float)w/(d_out_new_s*csr_in_deg[n]));
      float log_w = __log2f(w);
      float log_deg_in = __log2f(csr_in_deg[n]);
      float p = log_w - ( log_deg_in + log_d_out_new_s );
      dS -= (float)w*p;  
    }
    for (; i < ptr_r_e; i++) {
      w = csr_out_adj_wgt[i];
      n = csr_out_adj_node[i];
      //dS -= w * __log2f((float)w/(d_out_new_s*csr_in_deg[n]));
      float log_w = __log2f(w);
      float log_deg_in = __log2f(csr_in_deg[n]);
      float p = log_w - ( log_deg_in + log_d_out_new_s );
      dS -= (float)w*p;
    }
    for (; j < ptr_s_e; j++) {
      w = csr_out_adj_wgt[j];
      n = csr_out_adj_node[j];
      //dS -= w * __log2f((float)w/(d_out_new_s*csr_out_deg[n]));
      float log_w = __log2f(w);
      float log_deg_in = __log2f(csr_in_deg[n]);
      float p = log_w - ( log_deg_in + log_d_out_new_s );
      dS -= (float)w*p;  
    }
    dS_new_out[tid] = dS;
  }
}

template<typename T, typename V, typename W>
__global__ void calculate_dS_new_in(
  float* dS_new_in,
  V* proposed_blocks,
  T* csr_in_adj_ptr,
  V* csr_in_adj_node,
  W* csr_in_adj_wgt,
  W* csr_in_deg,
  W* csr_out_deg,
  int B,
  size_t csr_in_size,
  int num_proposals
) {

  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < B*num_proposals) {
    V r = tid / num_proposals;
    V s = proposed_blocks[tid];
    T ptr_r_s, ptr_r_e;
    T ptr_s_s, ptr_s_e;
    ptr_r_s = csr_in_adj_ptr[r];
    ptr_s_s = csr_in_adj_ptr[s];
    if (r + 1 < B) {
      ptr_r_e = csr_in_adj_ptr[r+1];
    }
    else {
      ptr_r_e = csr_in_size;
    }
    if (s + 1 < B) {
      ptr_s_e = csr_in_adj_ptr[s+1];
    }
    else {
      ptr_s_e = csr_in_size;
    }
    T i = ptr_r_s, j = ptr_s_s;
    float dS = 0;
    W w;
    W d_in_out_s = csr_in_deg[s] + csr_in_deg[r];
    float log_d_in_out_s = __log2f(d_in_out_s);
    V n, n1, n2;
    while (i < ptr_r_e && j < ptr_s_e) {
      n1 = csr_in_adj_node[i];
      n2 = csr_in_adj_node[j];
      if (n1 < n2) {
        w = csr_in_adj_wgt[i];
        n = n1;
        i++;
      }
      else if (n1 > n2) {
        w = csr_in_adj_wgt[j];
        n = n2;
        j++;
      }
      else {
        w = csr_in_adj_wgt[i] + csr_in_adj_wgt[j];
        n = n1;
        i++;
        j++;
      }
      if (n != r && n != s) {
        float log_w = __log2f(w);
        float log_deg_out = __log2f(csr_out_deg[n]);
        float p = log_w - ( log_deg_out + log_d_in_out_s );
        dS -= (float)w*p;
        //dS -= w * __log2f((float)w/(csr_out_deg[n]*d_in_out_s));
      }
    }
    for (; i < ptr_r_e; i++) {
      n = csr_in_adj_node[i];
      if (n != r && n != s) {
        w = csr_in_adj_wgt[i];
        float log_w = __log2f(w);
        float log_deg_out = __log2f(csr_out_deg[n]);
        float p = log_w - ( log_deg_out + log_d_in_out_s );
        dS -= (float)w*p;
        //dS -= w * __log2f((float)w/(csr_out_deg[n]*d_in_out_s));
      }
    }
    for (; j < ptr_s_e; j++) {
      n = csr_in_adj_node[j];
      if (n != r && n != s) {
        w = csr_in_adj_wgt[j];
        float log_w = __log2f(w);
        float log_deg_out = __log2f(csr_out_deg[n]);
        float p = log_w - ( log_deg_out + log_d_in_out_s );
        dS -= (float)w*p; 
        //dS -= w * __log2f((float)w/(csr_out_deg[n]*d_in_out_s));
      }
    }
    dS_new_in[tid] = dS;
  }
}


template<typename T, typename V, typename W>
__global__ void calculate_dS_overall(
  float* dS, 
  float* dS_out, 
  float* dS_in,
  float* dS_new_out, 
  float* dS_new_in,
  T* csr_out_adj_ptr,
  V* csr_out_adj_node,
  W* csr_out_adj_wgt,
  W* csr_out_deg,
  W* csr_in_deg,
  V* proposed_blocks,
  V* bestS,
  int B,
  int num_proposals
) {

  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < B*num_proposals) {
    V r = tid / num_proposals;
    V s = proposed_blocks[tid];
    V idx = tid % num_proposals;
    V sid = s*num_proposals + idx;
    float deltaS = 0;
    deltaS += dS_out[r] + dS_out[s];
    deltaS += dS_in[tid] + dS_in[sid];
    deltaS += dS_new_out[tid];
    deltaS += dS_new_in[tid];
    dS[tid] = deltaS;
  }
}


template<typename V, typename W>
__global__ void find_best_warp(
  float* dS,
  float* dS2,
  V* proposed_blocks,
  V* bestS,
  int B,
  int num_proposals
){
  
  // a warp is responsible for find best in num propose
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < B*warpSize) {
    unsigned warpid = tid / warpSize;
    unsigned tid_in_warp = threadIdx.x % warpSize;
    unsigned load_index = warpid * num_proposals + tid_in_warp;

    float ds2;
    V best;
    if (tid_in_warp < num_proposals) {
      ds2 = dS2[load_index];
      best = proposed_blocks[load_index];
    }
    else {
      ds2 = 3.402823466e+38F;
      best = 0;
    }
    
    for (unsigned offset = warpSize / 2; offset > 0; offset /= 2) {
      float other_ds2 = __shfl_down_sync(0xffffffff, ds2, offset);
      V other_best = __shfl_down_sync(0xffffffff, best, offset);
      ds2 = min(ds2, other_ds2);
      if (ds2 == other_ds2) best = other_best;
    }
    
    dS[warpid] = ds2;
    bestS[warpid] = best; 
  } 
}


template<typename T, typename V, typename W>
__global__ void calculate_dS_in_nodal_r(
  float* dS_in_r,
  T* csr_in_adj_ptr,
  V* csr_in_adj_node,
  W* csr_in_adj_wgt,
  W* csr_in_deg,
  W* csr_out_deg,
  V* original_blocks,
  V* proposed_blocks,
  int BS,
  int itr,
  int B,
  int N,
  size_t csr_in_size
) {

  unsigned ii = blockIdx.x * blockDim.x + threadIdx.x;

  if (ii < BS) {
    unsigned ni = ii + itr*BS;
    V r = original_blocks[ni];
    V s = proposed_blocks[ni];
    T ptr_s = csr_in_adj_ptr[r];
    T ptr_e;
    if (r + 1 < B) {
      ptr_e = csr_in_adj_ptr[r+1];
    }
    else {
      ptr_e = csr_in_size;
    }
    float dS = 0;
    float log_csr_in_deg_r = __log2f(csr_in_deg[r]);
    for (T i = ptr_s; i < ptr_e; i++) {
      V node = csr_in_adj_node[i];
      if (node != r && node != s) {
        W w = csr_in_adj_wgt[i];
        float log_w = __log2f(w);
        W csr_out_deg_i = csr_out_deg[node];
        float log_deg = __log2f(csr_out_deg_i);
        float p = log_w - ( log_deg + log_csr_in_deg_r );
        dS += (float)w*p;
      }
      //if (csr_in_adj_node[i] != r && csr_in_adj_node[i] != s) {
      //  dS += (float)csr_in_adj_wgt[i] * __log2f((float)csr_in_adj_wgt[i]
      //    / (csr_out_deg[csr_in_adj_node[i]] * csr_in_deg[r]));
      //}
    }
    dS_in_r[ii] = dS;
  }
}

template<typename T, typename V, typename W>
__global__ void calculate_dS_in_nodal_s(
  float* dS_in_s,
  T* csr_in_adj_ptr,
  V* csr_in_adj_node,
  W* csr_in_adj_wgt,
  W* csr_in_deg,
  W* csr_out_deg,
  V* original_blocks,
  V* proposed_blocks,
  int BS,
  int itr,
  int B,
  int N,
  size_t csr_in_size
) {


  unsigned ii = blockIdx.x * blockDim.x + threadIdx.x;

  if (ii < BS) {
    unsigned ni = ii + itr*BS;
    V r = original_blocks[ni];
    V s = proposed_blocks[ni];
    T ptr_s = csr_in_adj_ptr[s];
    T ptr_e;
    if (s + 1 < B) {
      ptr_e = csr_in_adj_ptr[s+1];
    }
    else {
      ptr_e = csr_in_size;
    }
    float dS = 0;
    float log_csr_in_deg_s = __log2f(csr_in_deg[s]);
    for (T i = ptr_s; i < ptr_e; i++) {
      V node = csr_in_adj_node[i];
      if (node != r && node != s) {
        W w = csr_in_adj_wgt[i];
        float log_w = __log2f(w);
        W csr_out_deg_i = csr_out_deg[node];
        float log_deg = __log2f(csr_out_deg_i);
        float p = log_w - ( log_deg + log_csr_in_deg_s );
        dS += (float)w*p;
      }
      //if (csr_in_adj_node[i] != r && csr_in_adj_node[i] != s) {
      //  dS += (float)csr_in_adj_wgt[i] * __log2f((float)csr_in_adj_wgt[i]
      //    / (csr_out_deg[csr_in_adj_node[i]] * csr_in_deg[s]));
      //}
    }
    dS_in_s[ii] = dS;
  }
}


template<typename V, typename W>
__global__ void calculate_dS_overall_nodal(
  float* dS,
  float* dS_new_r_in,
  float* dS_new_s_in,
  float* dS_new_s_out,
  float* dS_new_r_out,
  float* dS_in_r,
  float* dS_in_s,
  float* dS_out,
  V* original_blocks,
  V* proposed_blocks,
  int BS,
  int itr,
  int B,
  int N
) {


  unsigned ii = blockIdx.x * blockDim.x + threadIdx.x;

  if (ii < BS) {
    unsigned ni = ii + itr*BS;
    float deltaS = 0.f;
    V r = original_blocks[ni];
    V s = proposed_blocks[ni];
    if (r != s) {
      deltaS += dS_out[r] + dS_out[s];
      deltaS += dS_in_r[ii] + dS_in_s[ii];
      deltaS -= dS_new_r_out[ii];
      deltaS -= dS_new_s_out[ii];
      deltaS -= dS_new_r_in[ii];
      deltaS -= dS_new_s_in[ii];
    }
    if (deltaS < 0.f) {
      original_blocks[ni] = s;
      dS[ii] = deltaS;
    }
    else {
      dS[ii] = 0;
    }
  }

}

template<typename V>
__global__ void random_int_generator(
  V* gpu_random_blocks, 
  int numProposals, 
  int B
) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < B*numProposals) {
    curandState state;
    curand_init(clock64(), idx, 0, &state);
    gpu_random_blocks[idx] = curand(&state) % B;
  }

}

__global__ void uniform_number_generator(
  float* gpu_uniform_x, 
  int numProposals, 
  int B
) {

  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < B*numProposals) {
    curandState state;
    curand_init(clock64(), idx, 0, &state);
    gpu_uniform_x[idx] = curand_uniform(&state);
  }

}

template<typename V>
__global__ void uniform_number_generator2(
  float* gpu_uniform_x1,
  float* gpu_uniform_x2,
  float* gpu_uniform_x3,
  size_t size1,
  size_t size2,
  size_t size3,
  size_t max_size,
  V* gpu_random_blocks,
  size_t size4,
  size_t B 
) {

  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid < max_size) {
    curandState state;
    curand_init(clock64(), tid, 0, &state);
    if (tid < size1) 
      gpu_uniform_x1[tid] = curand_uniform(&state);
    if (tid < size2)
      gpu_uniform_x2[tid] = curand_uniform(&state);
    if (tid < size3)
      gpu_uniform_x3[tid] = curand_uniform(&state);
    if (tid < size4)
      gpu_random_blocks[tid] = curand(&state) % B;
  }
  
}

template<typename W>
__global__ void calculate_accepted_prob(
  float* gpu_accepted_prob,
  W* csr_out_deg,
  W* csr_in_deg,
  int B) {

  int idx =  blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < B) {
    W deg = csr_out_deg[idx] + csr_in_deg[idx];
    gpu_accepted_prob[idx] = (float)B/(deg+B);
  }

}


template<typename T, typename V, typename W>
__global__ void choose_neighbors(
  float* gpu_uniform_x1,
  float* gpu_uniform_x2,
  T* csr_out_adj_ptr,
  V* csr_out_adj_node,
  size_t csr_out_adj_node_size,
  W* csr_out_adj_wgt,
  W* csr_out_deg,
  T* csr_in_adj_ptr,
  V* csr_in_adj_node,
  size_t csr_in_adj_node_size,
  W* csr_in_adj_wgt,
  W* csr_in_deg,
  V* sampling_neighbor_u,
  V* sampling_neighbor_s,
  int numProposals,
  int B
) {

  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < B*numProposals) {
    unsigned r = idx / numProposals;
    V u = 0;
    float p = gpu_uniform_x1[idx];
    float ps = 0;
    T out_ptr_begin = csr_out_adj_ptr[r];
    T in_ptr_begin = csr_in_adj_ptr[r];
    T out_ptr_end;
    T in_ptr_end;
    if (r + 1 < B) {
      out_ptr_end = csr_out_adj_ptr[r+1];
      in_ptr_end = csr_in_adj_ptr[r+1];
    }
    else {
      out_ptr_end = csr_out_adj_node_size;
      in_ptr_end = csr_in_adj_node_size;
    }
    W deg = csr_out_deg[r] + csr_in_deg[r];
    int find = 0;
    for (T i = out_ptr_begin; i < out_ptr_end; i++) {
      ps += (float)csr_out_adj_wgt[i]/deg;
      if (ps >= p) {
        u = csr_out_adj_node[i];
        find = 1;
        break;
      }
    }
    if (find == 0) {
      for (T i = in_ptr_begin; i < in_ptr_end; i++) {
        ps += (float)csr_in_adj_wgt[i]/deg;
        if (ps >= p) {
          u = csr_in_adj_node[i];
          break;
        }
      }
    }
    sampling_neighbor_u[idx] = u;
    p = gpu_uniform_x2[idx];
    V s = 0;
    ps = 0;
    find = 0;
    out_ptr_begin = csr_out_adj_ptr[u];
    in_ptr_begin = csr_in_adj_ptr[u];
    if (u + 1 < B) {
      out_ptr_end = csr_out_adj_ptr[u+1];
      in_ptr_end = csr_in_adj_ptr[u+1];
    }
    else {
      out_ptr_end = csr_out_adj_node_size;
      in_ptr_end = csr_in_adj_node_size;
    }
    W deg_u = csr_out_deg[u] + csr_in_deg[u];
    for (T i = out_ptr_begin; i < out_ptr_end; i++) {
      ps += (float)csr_out_adj_wgt[i]/deg_u;
      if (ps >= p && csr_out_adj_node[i] != r) {
        s = csr_out_adj_node[i];
        find = 1;
        break;
      }
    }
    if (find == 0) {
      for (T i = in_ptr_begin; i < in_ptr_end; i++) {
        ps += (float)csr_in_adj_wgt[i]/deg_u;
        if (ps >= p && csr_in_adj_node[i] != r) {
          s = csr_in_adj_node[i];
          break;
        }
      }
    }
    sampling_neighbor_s[idx] = s;
  }

}


template<typename V, typename W>
__global__ void propose_blocks(
  V* gpu_random_blocks,
  float* gpu_uniform_x,
  float* gpu_accepted_prob,
  V* sampling_neighbor_u,
  V* sampling_neighbor_s,
  int numProposals,
  int B,
  W* csr_out_deg,
  W* csr_in_deg,
  V* S
) {

  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < B*numProposals) {
    unsigned r = idx / numProposals;
    W deg = csr_out_deg[r] + csr_in_deg[r];
    if (deg == 0) {
      S[idx] = gpu_random_blocks[idx];
    }
    else {
      V u = sampling_neighbor_u[idx];
      if (gpu_uniform_x[idx] <= gpu_accepted_prob[u]) {
        S[idx] = gpu_random_blocks[idx];
      }
      else {
        S[idx] = sampling_neighbor_s[idx];
      }
    }
  }

}


template<typename V>
__global__ void random_int_generator_nodal(
  V* gpu_random_blocks, 
  int B, 
  int BS
) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < BS) {
    curandState state;
    curand_init(clock64(), idx, 0, &state);
    gpu_random_blocks[idx] = curand(&state) % B;
  }

}


template<typename T, typename V, typename W>
__global__ void choose_neighbored_node_block(
  float* gpu_uniform_x,
  T* g_csr_out_adj_ptr,
  V* g_csr_out_adj_node,
  size_t g_csr_out_adj_node_size,
  W* g_csr_out_adj_wgt,
  W* node_deg_out,
  T* g_csr_in_adj_ptr,
  V* g_csr_in_adj_node,
  size_t g_csr_in_adj_node_size,
  W* g_csr_in_adj_wgt,
  W* node_deg_in,
  V* partitions,
  V* sampling_neighbor_u,
  int itr,
  int BS,
  int N
){

  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < BS) {
    unsigned ni = idx + itr*BS;
    float p = gpu_uniform_x[idx];
    float ps = 0;
    V u_node = 0;
    T ptr_begin_out, ptr_end_out;
    T ptr_begin_in, ptr_end_in;
    ptr_begin_out = g_csr_out_adj_ptr[ni];
    ptr_begin_in= g_csr_in_adj_ptr[ni];
    if (ni + 1 < N) {
      ptr_end_out = g_csr_out_adj_ptr[ni+1];
      ptr_end_in = g_csr_in_adj_ptr[ni+1];
    }
    else {
      ptr_end_out = g_csr_out_adj_node_size;
      ptr_end_in = g_csr_in_adj_node_size;
    }
    W deg = node_deg_out[ni] + node_deg_in[ni];
    int find = 0;
    for (T i = ptr_begin_out; i < ptr_end_out; i++) {
      ps += (float)g_csr_out_adj_wgt[i]/deg;
      if (ps >= p) {
        u_node = g_csr_out_adj_node[i];
        find = 1;
        break;
      }
    }
    if (find == 0) {
      for (T i = ptr_begin_in; i < ptr_end_in; i++) {
        ps += (float)g_csr_in_adj_wgt[i]/deg;
        if (ps >= p) {
          u_node = g_csr_in_adj_node[i];
          break;
        }
      }
    }
    V u = partitions[u_node];
    sampling_neighbor_u[idx] = u;
  }

}


template<typename T, typename V, typename W>
__global__ void choose_neighbored_block(
  float* gpu_uniform_x,
  T* csr_out_adj_ptr,
  V* csr_out_adj_node,
  size_t csr_out_adj_node_size,
  W* csr_out_adj_wgt,
  W* csr_out_deg,
  T* csr_in_adj_ptr,
  V* csr_in_adj_node,
  size_t csr_in_adj_node_size,
  W* csr_in_adj_wgt,
  W* csr_in_deg,
  V* sampling_neighbor_s,
  int B
) {

  unsigned r = blockIdx.x * blockDim.x + threadIdx.x;

  if (r < B) {
    float p = gpu_uniform_x[r];
    V u = 0;
    float ps = 0;
    T ptr_begin_out = csr_out_adj_ptr[r];
    T ptr_begin_in = csr_in_adj_ptr[r];
    T ptr_end_out, ptr_end_in;
    if (r + 1 < B) {
      ptr_end_out = csr_out_adj_ptr[r+1];
      ptr_end_in = csr_in_adj_ptr[r+1];
    }
    else {
      ptr_end_out = csr_out_adj_node_size;
      ptr_end_in = csr_in_adj_node_size;
    }
    int find = 0;
    W deg = csr_out_deg[r] + csr_in_deg[r];
    for (T i = ptr_begin_out; i < ptr_end_out; i++) {
      ps += (float)csr_out_adj_wgt[i]/deg;
      if (ps >= p) {
        u = csr_out_adj_node[i];
        find = 1;
        break;
      }
    }
    if (find == 0) {
      for (T i = ptr_begin_in; i < ptr_end_in; i++) {
        ps += (float)csr_in_adj_wgt[i]/deg;
        if (ps >= p) {
          u = csr_in_adj_node[i];
          break;
        }
      }
    }
    sampling_neighbor_s[r] = u;
  }
}


template<typename V, typename W>
__global__ void propose_nodal(
  V* gpu_random_blocks,
  float* gpu_uniform_x,
  float* gpu_accepted_prob,
  W* node_deg_out,
  W* node_deg_in,
  V* sampling_neighbor_u,
  V* sampling_neighbor_s,
  int BS,
  int itr,
  V* S
) {

  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < BS) {
    unsigned ni = idx + itr*BS;
    W deg = node_deg_out[ni] + node_deg_in[ni];
    if (deg == 0) {
      S[ni] = gpu_random_blocks[idx];
    }
    else {
      V u = sampling_neighbor_u[idx];
      if (gpu_uniform_x[u] <= gpu_accepted_prob[u]) {
        S[ni] = gpu_random_blocks[idx];
      }
      else {
        S[ni] = sampling_neighbor_s[u];
      }
    }
  }
}

template<typename T, typename V, typename W>
__global__ void block_deg_mapping(
  T* csr_out_adj_ptr,
  V* csr_out_adj_node,
  size_t csr_out_adj_node_size,
  W* csr_out_adj_wgt,
  W* csr_out_adj_node_deg_out,
  W* csr_out_adj_node_deg_in,
  W* csr_out_deg,
  W* csr_in_deg,
  int B
) {

  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < B) {
    T ptr_begin = csr_out_adj_ptr[idx];
    T ptr_end;
    if (idx + 1 < B) {
      ptr_end = csr_out_adj_ptr[idx+1];
    }
    else {
      ptr_end = csr_out_adj_node_size;
    }
    for (T i = ptr_begin; i < ptr_end; i++) {
      csr_out_adj_node_deg_out[i] = csr_out_deg[idx];
      csr_out_adj_node_deg_in[i] = csr_in_deg[csr_out_adj_node[i]];
    }
  }
}


template<typename T, typename V, typename W>
__global__ void compute_overall_S_transform(
  T* csr_out_adj_ptr,
  V* csr_out_adj_node,
  size_t csr_out_adj_node_size,
  W* csr_out_adj_wgt,
  float* dataS,
  W* csr_out_adj_node_deg_out,
  W* csr_out_adj_node_deg_in,
  int B
) {

  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < csr_out_adj_node_size) {
    W w = csr_out_adj_wgt[i];
    W deg_out = csr_out_adj_node_deg_out[i];
    W deg_in = csr_out_adj_node_deg_in[i];
    float log_w = __log2f(w);
    float log_deg_out = __log2f(deg_out);
    float log_deg_in = __log2f(deg_in);
    float p = log_w - (log_deg_out + log_deg_in);
    dataS[i] = (float)w*p;
    //dataS[i] = (float)csr_out_adj_wgt[i] * __log2f(
    //  (float)csr_out_adj_wgt[i]/
    //  (csr_out_adj_node_deg_out[i]*csr_out_adj_node_deg_in[i]));
  }
}


template<typename T, typename V, typename W>
__global__ void match_neighbored_block_out(
  T* g_csr_out_adj_ptr,
  V* g_csr_out_adj_node,
  size_t g_csr_out_adj_node_size,
  V* g_csr_out_adj_block,
  V* partitions,
  int N
) {

  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    T ptr_begin = g_csr_out_adj_ptr[idx];
    T ptr_end;
    if (idx + 1 < N) {
      ptr_end = g_csr_out_adj_ptr[idx+1];
    }
    else {
      ptr_end = g_csr_out_adj_node_size;
    }
    for (T i = ptr_begin; i < ptr_end; i++) {
      g_csr_out_adj_block[i] = partitions[g_csr_out_adj_node[i]];
    }
  }

}


template<typename T, typename V, typename W>
__global__ void match_neighbored_block_in(
  T* g_csr_in_adj_ptr,
  V* g_csr_in_adj_node,
  size_t g_csr_in_adj_node_size,
  V* g_csr_in_adj_block,
  V* partitions,
  int N
) {

  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    T ptr_begin = g_csr_in_adj_ptr[idx];
    T ptr_end;
    if (idx + 1 < N) {
      ptr_end = g_csr_in_adj_ptr[idx+1];
    }
    else {
      ptr_end = g_csr_in_adj_node_size;
    }
    for (T i = ptr_begin; i < ptr_end; i++) {
      g_csr_in_adj_block[i] = partitions[g_csr_in_adj_node[i]];
    }
  }

}

template<typename T, typename V, typename W>
__global__ void calculate_dS_new_r_in_nodal_new(
  float* dS_new_r_in,
  V* original_blocks,
  V* proposed_blocks,
  T* csr_in_adj_ptr,
  V* csr_in_adj_node,
  size_t csr_in_adj_node_size,
  W* csr_in_adj_wgt,
  T* g_csr_in_adj_ptr,
  V* g_csr_in_adj_block,
  size_t g_csr_in_adj_block_size,
  W* g_csr_in_adj_wgt,
  W* node_deg_in,
  W* csr_in_deg,
  W* csr_out_deg,
  int BS,
  int itr,
  int B,
  int N
) {

  unsigned ii = blockIdx.x * blockDim.x + threadIdx.x;

  if (ii < BS) {
    unsigned ni = ii + itr*BS;
    V r = original_blocks[ni];
    V s = proposed_blocks[ni];
    float dS = 0;
    W k_in = node_deg_in[ni];
    W d_in_new_r = csr_in_deg[r] - k_in;
    float log_d_in_new_r = __log2f(d_in_new_r);

    T ptr_begin_block = csr_in_adj_ptr[r];
    T ptr_end_block;
    if (r + 1 < B) {
      ptr_end_block = csr_in_adj_ptr[r+1];
    }
    else {
      ptr_end_block = csr_in_adj_node_size;
    }
    T ptr_begin_node = g_csr_in_adj_ptr[ni];
    T ptr_end_node;
    if (ni + 1 < N) {
      ptr_end_node = g_csr_in_adj_ptr[ni+1];
    }
    else {
      ptr_end_node = g_csr_in_adj_block_size;
    }

    T i = ptr_begin_block, j = ptr_begin_node;
    W w = 0;
    V n, n1, n2;
    while (i < ptr_end_block && j < ptr_end_node) {
      n1 = csr_in_adj_node[i];
      n2 = g_csr_in_adj_block[j];
      if (n1 < n2) {
        w = csr_in_adj_wgt[i];
        n = n1;
        i++;
      }
      else if (n1 > n2) {
        w = g_csr_in_adj_wgt[j];
        n = n2;
        j++;
      }
      else {
        w = csr_in_adj_wgt[i] - g_csr_in_adj_wgt[j];
        n = n1;
        i++;
        j++;
        while (n == g_csr_in_adj_block[j] && j < ptr_end_node) {
          w -= g_csr_in_adj_wgt[j];
          j++;
        }
      }
      if (n != r && n != s && w != 0) {
        //dS += w * __log2f((float)w/(csr_out_deg[n]*d_in_new_r));
        float log_csr_out_deg_n = __log2f(csr_out_deg[n]);
        float log_w = __log2f(w);
        float p = log_w - ( log_csr_out_deg_n + log_d_in_new_r);
        dS += (float)w*p;
      }
    }
    for (; i < ptr_end_block; i++) {
      w = csr_in_adj_wgt[i];
      n = csr_in_adj_node[i];
      if (n != r && n != s) {
        //dS += w * __log2f((float)w/(csr_out_deg[n]*d_in_new_r));
        float log_csr_out_deg_n = __log2f(csr_out_deg[n]);
        float log_w = __log2f(w);
        float p = log_w - ( log_csr_out_deg_n + log_d_in_new_r);
        dS += (float)w*p;
      }
    }
    dS_new_r_in[ii] = dS;
  }
}


template<typename T, typename V, typename W>
__global__ void calculate_dS_new_s_in_nodal_new(
  float* dS_new_s_in,
  V* original_blocks,
  V* proposed_blocks,
  T* csr_in_adj_ptr,
  V* csr_in_adj_node,
  size_t csr_in_adj_node_size,
  W* csr_in_adj_wgt,
  T* g_csr_in_adj_ptr,
  V* g_csr_in_adj_block,
  size_t g_csr_in_adj_block_size,
  W* g_csr_in_adj_wgt,
  W* node_deg_in,
  W* csr_in_deg,
  W* csr_out_deg,
  int BS,
  int itr,
  int B,
  int N
) {

  unsigned ii = blockIdx.x * blockDim.x + threadIdx.x;

  if (ii < BS) {
    unsigned ni = ii + itr*BS;
    V r = original_blocks[ni];
    V s = proposed_blocks[ni];
    float dS = 0;
    W k_in = node_deg_in[ni];
    W d_in_new_s = csr_in_deg[s] + k_in;
    float log_d_in_new_s = __log2f(d_in_new_s);
    
    T ptr_begin_block = csr_in_adj_ptr[s];
    T ptr_end_block;
    if (s + 1 < B) {
      ptr_end_block = csr_in_adj_ptr[s+1];
    }
    else {
      ptr_end_block = csr_in_adj_node_size;
    }
    T ptr_begin_node = g_csr_in_adj_ptr[ni];
    T ptr_end_node;
    if (ni + 1 < N) {
      ptr_end_node = g_csr_in_adj_ptr[ni+1];
    }
    else {
      ptr_end_node = g_csr_in_adj_block_size;
    }
    
    T i = ptr_begin_block, j = ptr_begin_node;
    W w = 0;
    V n, n1, n2;
    while (i < ptr_end_block && j < ptr_end_node) {
      n1 = csr_in_adj_node[i];
      n2 = g_csr_in_adj_block[j];
      if (n1 < n2) {
        w = csr_in_adj_wgt[i];
        n = n1;
        i++;
      }
      else if (n1 > n2) {
        w = g_csr_in_adj_wgt[j];
        n = n2;
        j++;
      }
      else {
        w = csr_in_adj_wgt[i] + g_csr_in_adj_wgt[j];
        n = n1;
        i++;
        j++;
        while (n == g_csr_in_adj_block[j] && j < ptr_end_node) {
          w += g_csr_in_adj_wgt[j];
          j++;
        }
      }
      if (n != r && n != s && w != 0) {
        //dS += w * __log2f((float)w/(csr_out_deg[n]*d_in_new_s));
        float log_w = __log2f(w);
        float log_deg_out = __log2f(csr_out_deg[n]);
        float p = log_w - ( log_deg_out + log_d_in_new_s );
        dS += (float)w*p;
      }
    }
    for (; i < ptr_end_block; i++) {
      n = csr_in_adj_node[i];
      if (n != r && n != s) {
        w = csr_in_adj_wgt[i];
        float log_w = __log2f(w);
        float log_deg_out = __log2f(csr_out_deg[n]);
        float p = log_w - ( log_deg_out + log_d_in_new_s );
        dS += (float)w*p;
        //dS += w * __log2f((float)w/(csr_out_deg[n]*d_in_new_s));
      }
    }
    for (; j < ptr_end_node; j++) {
      n = g_csr_in_adj_block[j];
      if (n != r && n != s) {
        w = csr_in_adj_wgt[i];
        float log_w = __log2f(w);
        float log_deg_out = __log2f(csr_out_deg[n]);
        float p = log_w - ( log_deg_out + log_d_in_new_s );
        dS += (float)w*p;
        //dS += w * __log2f((float)w/(csr_out_deg[n]*d_in_new_s));
      }
    }
    dS_new_s_in[ii] = dS;
  }

}



template<typename T, typename V, typename W>
__global__ void find_csr_in_r_wgt(
  W* g_csr_in_r_wgt,
  V* original_blocks,
  T* g_csr_in_adj_ptr,
  V* g_csr_in_adj_block,
  size_t g_csr_in_adj_block_size,
  W* g_csr_in_adj_wgt,
  int BS,
  int itr,
  int B,
  int N
) {

  unsigned ii = blockIdx.x * blockDim.x + threadIdx.x;

  if (ii < BS) {
    unsigned ni = ii + itr*BS;
    V r = original_blocks[ni];
    T ptr_begin, ptr_end;
    ptr_begin = g_csr_in_adj_ptr[ni];
    if (ni + 1 < N) {
      ptr_end = g_csr_in_adj_ptr[ni+1];
    }
    else {
      ptr_end = g_csr_in_adj_block_size;
    }
    W w = 0;
    for (T i = ptr_begin; i < ptr_end; i++) {
      if (g_csr_in_adj_block[i] == r) {
        w += g_csr_in_adj_wgt[i];
      }
    }
    g_csr_in_r_wgt[ni] = w;
  }

}


template<typename T, typename V, typename W>
__global__ void find_csr_in_s_wgt(
  W* g_csr_in_s_wgt,
  V* proposed_blocks,
  T* g_csr_in_adj_ptr,
  V* g_csr_in_adj_block,
  size_t g_csr_in_adj_block_size,
  W* g_csr_in_adj_wgt,
  int BS,
  int itr,
  int B,
  int N
) {

  unsigned ii = blockIdx.x * blockDim.x + threadIdx.x;

  if (ii < BS) {
    unsigned ni = ii + itr*BS;
    V s = proposed_blocks[ni];
    T ptr_begin, ptr_end;
    ptr_begin = g_csr_in_adj_ptr[ni];
    if (ni + 1 < N) {
      ptr_end = g_csr_in_adj_ptr[ni+1];
    }
    else {
      ptr_end = g_csr_in_adj_block_size;
    }
    W w = 0;
    for (T i = ptr_begin; i < ptr_end; i++) {
      if (g_csr_in_adj_block[i] == s) {
        w += g_csr_in_adj_wgt[i];
      }
    }
    g_csr_in_s_wgt[ni] = w;
  }

}


template<typename T, typename V, typename W>
__global__ void calculate_dS_new_r_out_nodal_new(
  float* dS_new_r_out,
  V* original_blocks,
  V* proposed_blocks,
  T* csr_out_adj_ptr,
  V* csr_out_adj_node,
  size_t csr_out_adj_node_size,
  W* csr_out_adj_wgt,
  T* g_csr_out_adj_ptr,
  V* g_csr_out_adj_block,
  size_t g_csr_out_adj_block_size,
  W* g_csr_out_adj_wgt,
  W* node_deg_out,
  W* csr_in_deg,
  W* csr_out_deg,
  W* g_csr_in_r_wgt,
  int BS,
  int itr,
  int B,
  int N
) {

  unsigned ii = blockIdx.x * blockDim.x + threadIdx.x;

  if (ii < BS) {
    unsigned ni = ii + itr*BS;
    V r = original_blocks[ni];
    V s = proposed_blocks[ni];
    float dS = 0;
    W k_out = node_deg_out[ni];
    W d_out_new_r = csr_out_deg[r] - k_out;
    float log_d_out_new_r = __log2f(d_out_new_r);
    T ptr_begin_block, ptr_end_block;
    T ptr_begin_node, ptr_end_node;
    ptr_begin_block = csr_out_adj_ptr[r];
    if (r + 1 < B) {
      ptr_end_block = csr_out_adj_ptr[r+1];
    }
    else {
      ptr_end_block = csr_out_adj_node_size;
    }
    ptr_begin_node = g_csr_out_adj_ptr[ni];
    if (ni + 1 < N) {
      ptr_end_node = g_csr_out_adj_ptr[ni+1];
    }
    else {
      ptr_end_node = g_csr_out_adj_block_size;
    }

    T i = ptr_begin_block, j = ptr_begin_node;
    W w = 0;
    W ni_w = g_csr_in_r_wgt[ni];
    V n, n1, n2;
    bool find = 0;
    while (i < ptr_end_block && j < ptr_end_node) {
      n1 = csr_out_adj_node[i];
      n2 = g_csr_out_adj_block[j];
      if (n1 < n2) {
        w = csr_out_adj_wgt[i];
        n = n1;
        i++;
        if (n == r) {
          w -= ni_w;
        }
        if (n == s) {
          w += ni_w;
          find = 1;
        }
        if (w > 0) {
          //dS += w * __log2f((float)w/(csr_in_deg[n]*d_out_new_r));
          float log_w = __log2f(w);
          float log_csr_in_deg = __log2f(csr_in_deg[n]);
          float p = log_w - ( log_csr_in_deg + log_d_out_new_r );
          dS += (float)w*p;
        }
      }
      else if (n1 > n2) {
        w = g_csr_out_adj_wgt[j];
        n = n2;
        j++;
      }
      else {
        w = csr_out_adj_wgt[i] - g_csr_out_adj_wgt[j];
        n = n1;
        i++;
        j++;
        while (n == g_csr_out_adj_block[j] && j < ptr_end_node) {
          w -= g_csr_out_adj_wgt[j];
          j++;
        }
        if (n == r) {
          w -= ni_w;
        }
        if (n == s) {
          w += ni_w;
          find = 1;
        }
        if (w > 0) {
          //dS += w * __log2f((float)w/(csr_in_deg[n]*d_out_new_r));
          float log_w = __log2f(w);  
          float log_csr_in_deg = __log2f(csr_in_deg[n]);
          float p = log_w - ( log_csr_in_deg + log_d_out_new_r );
          dS += (float)w*p;
        }
      }
    }
    for (; i < ptr_end_block; i++) {
      w = csr_out_adj_wgt[i];
      n = csr_out_adj_node[i];
      if (n == r) {
        w -= g_csr_in_r_wgt[ni];
      }
      if (n == s) {
        w += g_csr_in_r_wgt[ni];
        find = 1;
      }
      if (w > 0) {
        //dS += w * __log2f((float)w/(csr_in_deg[n]*d_out_new_r));
        float log_w = __log2f(w);
        float log_csr_in_deg = __log2f(csr_in_deg[n]);
        float p = log_w - ( log_csr_in_deg + log_d_out_new_r );
        dS += (float)w*p;
      }
    }
    //if (find ==  0 && g_csr_in_r_wgt[ni] > 0) {
    //  w = g_csr_in_r_wgt[ni];
    //  dS += w * __log2f((float)w/(csr_in_deg[s]*d_out_new_r));
    //}
    if (find == 0 && ni_w > 0) {
      float log_ni_w = __log2f(ni_w);
      float log_csr_in_deg = __log2f(csr_in_deg[s]);
      float p = log_ni_w - (log_csr_in_deg + log_d_out_new_r);
      dS += (float)ni_w*p;
    }
    dS_new_r_out[ii] = dS;
  }
}


template<typename T, typename V, typename W>
__global__ void calculate_dS_new_s_out_nodal_new(
  float* dS_new_s_out,
  V* original_blocks,
  V* proposed_blocks,
  T* csr_out_adj_ptr,
  V* csr_out_adj_node,
  size_t csr_out_adj_node_size,
  W* csr_out_adj_wgt,
  T* g_csr_out_adj_ptr,
  V* g_csr_out_adj_block,
  size_t g_csr_out_adj_block_size,
  W* g_csr_out_adj_wgt,
  W* node_deg_out,
  W* csr_in_deg,
  W* csr_out_deg,
  W* g_csr_in_s_wgt,
  int BS,
  int itr,
  int B,
  int N
) {

  unsigned ii = blockIdx.x * blockDim.x + threadIdx.x;

  if (ii < BS) {
    unsigned ni = ii + itr*BS;
    V r = original_blocks[ni];
    V s = proposed_blocks[ni];
    float dS = 0;
    W k_out = node_deg_out[ni];
    W d_out_new_s = csr_out_deg[s] + k_out;
    float log_d_out_new_s = __log2f(d_out_new_s);
    
    T ptr_begin_block, ptr_end_block;
    T ptr_begin_node, ptr_end_node;
    ptr_begin_block = csr_out_adj_ptr[s];
    if (s + 1 < B) {
      ptr_end_block = csr_out_adj_ptr[s+1];
    }
    else {
      ptr_end_block = csr_out_adj_node_size;
    }
    ptr_begin_node = g_csr_out_adj_ptr[ni];
    if (ni + 1 < N) {
      ptr_end_node = g_csr_out_adj_ptr[ni+1];
    }
    else {
      ptr_end_node = g_csr_out_adj_block_size;
    }
    
    T i = ptr_begin_block, j = ptr_begin_node;
    W w = 0;
    W ni_w = g_csr_in_s_wgt[ni];
    V n, n1, n2;
    bool find = 0;
    while (i < ptr_end_block && j < ptr_end_node) {
      n1 = csr_out_adj_node[i];
      n2 = g_csr_out_adj_block[j];
      if (n1 < n2) {
        w = csr_out_adj_wgt[i];
        n = n1;
        i++;
      }
      else if (n1 > n2) {
        w = g_csr_out_adj_wgt[j];
        n = n2;
        j++;
      }
      else {
        w = csr_out_adj_wgt[i] + g_csr_out_adj_wgt[j];
        n = n1;
        i++;
        j++;
        while (n == g_csr_out_adj_block[j] && j < ptr_end_node) {
          w += g_csr_out_adj_wgt[j];
          j++;
        }
      }
      if (n == r) {
        w -= ni_w;
      }
      if (n == s) {
        w += ni_w;
        find = 1;
      }
      if (w != 0) {
        //dS += w * __log2f((float)w/(csr_in_deg[n]*d_out_new_s));
        float log_w = __log2f(w);
        float log_csr_in_deg = __log2f(csr_in_deg[n]);
        float p = log_w - ( log_csr_in_deg + log_d_out_new_s );
        dS += (float)w*p;
      }
    }
    for (; i < ptr_end_block; i++) {
      w = csr_out_adj_wgt[i];
      n = csr_out_adj_node[i];
      if (n == r) {
        w -= ni_w;
      }
      if (n == s) {
        w += ni_w;
        find = 1;
      }
      if (w != 0) {
        //dS += w * __log2f((float)w/(csr_in_deg[n]*d_out_new_s));
        float log_w = __log2f(w); 
        float log_csr_in_deg = __log2f(csr_in_deg[n]);
        float p = log_w - ( log_csr_in_deg + log_d_out_new_s );
        dS += (float)w*p;
      }
    }
    for (; j < ptr_end_node; j++) {
      w = g_csr_out_adj_wgt[j];
      n = g_csr_out_adj_block[j];
      if (n == r) {
        w -= ni_w;
      }
      if (n == s) {
        w += ni_w;
        find = 1;
      }
      if (w != 0) {
        //dS += w * __log2f((float)w/(csr_in_deg[n]*d_out_new_s));
        float log_w = __log2f(w);       
        float log_csr_in_deg = __log2f(csr_in_deg[n]);
        float p = log_w - ( log_csr_in_deg + log_d_out_new_s );
        dS += (float)w*p;
      }
    }
    //if (find ==  0 && g_csr_in_s_wgt[ni] > 0) {
    //  w = g_csr_in_s_wgt[ni];
    //  dS += w * __log2f((float)w/(csr_in_deg[s]*d_out_new_s));
    //}
    if (find == 0 && ni_w > 0) {
      float log_ni_w = __log2f(ni_w);
      float log_csr_in_deg = __log2f(csr_in_deg[s]);
      float p = log_ni_w - (log_csr_in_deg + log_d_out_new_s);
      dS += (float)ni_w*p;
    }
    dS_new_s_out[ii] = dS;
  }
}

template <typename V>  
__global__ void update_partitions(
  V* partitions,
  V* block_map,
  unsigned N
) {

  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) {
    partitions[tid] = block_map[partitions[tid]];
  }

}
