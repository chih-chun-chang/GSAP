#include <vector>
#include <string>
#include <fstream>
#include "graph.hpp"

template <typename T, typename V, typename W>
Graph<T, V, W> readFile(int numNodes) {
  
  std::string FileName("../Dataset/static/lowOverlap_lowBlockSizeVar/static_lowOverlap_lowBlockSizeVar");
  //std::string FileName("/home/chang292/uSAP/Dataset/static/lowOverlap_highBlockSizeVar/static_lowOverlap_highBlockSizeVar");
  //std::string FileName("/home/chang292/uSAP/Dataset/static/highOverlap_lowBlockSizeVar/static_highOverlap_lowBlockSizeVar");
  //std::string FileName("/home/chang292/uSAP/Dataset/static/highOverlap_highBlockSizeVar/static_highOverlap_highBlockSizeVar");


  switch(numNodes)  {
    case 1000:
      FileName += "_1000_nodes";
      break;
    case 5000:
      FileName += "_5000_nodes";
      break;
    case 20000:
      FileName += "_20000_nodes";
      break;
    case 50000:
      FileName += "_50000_nodes";
      break;
    case 200000:
      FileName += "_200000_nodes";
      break;
    case 1000000:
      FileName += "_1000000_nodes";
      break;
    case 5000000:
      FileName += "_5000000_nodes";
      break;
    case 20000000:
      FileName += "_20000000_nodes";
      break;
    default:
      std::cerr << "usage: ./run [Number of ints=1000/5000/20000/50000]\n";
      std::exit(1);
  }

  std::ifstream file(FileName + ".tsv");
  if (!file.is_open()) {
    std::cerr << "Unable to open file!\n";
    std::exit(EXIT_FAILURE);
  }

  Graph<T, V, W> g;

  g.N = 0; 

  std::string line; // format: node i \t node j \t  w_ij
  std::vector<std::string> v_line;
  V from, to;
  W weight;
  while (std::getline(file, line)) {
    unsigned start = 0; 
    unsigned tab_pos = line.find('\t');
    from = static_cast<V>(std::stoul(line.substr(start, tab_pos - start)));
    start = tab_pos + 1; 
    tab_pos = line.find('\t', start);
    to = static_cast<V>(std::stoul(line.substr(start, tab_pos - start)));
    start = tab_pos + 1; 
    tab_pos = line.find('\t', start);
    weight = static_cast<W>(std::stof(line.substr(start, tab_pos - start)));
    g.edges.emplace_back(from, to, weight);
    if (from > g.N) g.N = from;
  }
  file.close();

  g.E = g.edges.size();

  g.out_neighbors.resize(g.N);
  g.in_neighbors.resize(g.N);
  for (const auto& e : g.edges) {
    g.out_neighbors[e.from-1].emplace_back(e.to-1, e.wgt);
    g.in_neighbors[e.to-1].emplace_back(e.from-1, e.wgt);
  }

  // create Csr
  g.csr_out.adj_ptr.resize(g.N);
  g.csr_in.adj_ptr.resize(g.N);
  g.node_deg_out.resize(g.N);
  g.node_deg_in.resize(g.N);
  T ptr_out = 0;
  T ptr_in = 0;
  for (int i = 0; i < g.N; i++) {
    g.csr_out.adj_ptr[i] = ptr_out;
    g.csr_in.adj_ptr[i] = ptr_in;
    for (const auto& [v, w] : g.out_neighbors[i]) {
      g.csr_out.adj_node.emplace_back(v);
      g.csr_out.adj_wgt.emplace_back(w);
      g.node_deg_out[i] += w;
    }
    ptr_out += g.out_neighbors[i].size();
    for (const auto& [v, w] : g.in_neighbors[i]) {
      g.csr_in.adj_node.emplace_back(v);
      g.csr_in.adj_wgt.emplace_back(w);
      g.node_deg_in[i] += w;
    }
    ptr_in += g.in_neighbors[i].size();
  }


  // load the true partition
  std::ifstream true_file(FileName + "_truePartition.tsv");
  if (!true_file.is_open()) {
    std::cerr << "Unable to open file!\n";
    std::exit(EXIT_FAILURE);
  }
  // format: node i \t block
  while (std::getline(true_file, line)) {
    int start = 0;
    int tab_pos = line.find('\t');
    V i = std::stoi(line.substr(start, tab_pos - start));
    start = tab_pos + 1;
    tab_pos = line.find('\t', start);
    V block = static_cast<V>(std::stoul(line.substr(start, tab_pos - start)));
    g.truePartitions.emplace_back(block-1);
  }
  true_file.close();

  return g;
} // end of load_graph_from_tsv
