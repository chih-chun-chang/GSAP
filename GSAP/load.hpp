#include <vector>
#include <string>
#include <fstream>
#include "graph.hpp"

template <typename T, typename V, typename W>
Graph<T, V, W> readFile(std::string& ca, std::string& num) {
  
  std::string fileName("../Dataset/static/");

  if (ca == "LL") {
    fileName += "lowOverlap_lowBlockSizeVar/";
  }
  else if (ca == "LH") {
    fileName += "lowOverlap_highBlockSizeVar/";
  }
  else if (ca == "HL") {
    fileName += "highOverlap_lowBlockSizeVar/";
  }
  else if (ca == "HH") {
    fileName += "highOverlap_highBlockSizeVar/";
  }
  else {
    std::cerr << "usage: ./run [LL/LH/HL/HH] [1k/5k/20k/50k/200k/1m]\n";
    std::exit(1);
  }

  if (num == "1k") {
    fileName += "1k/1000_nodes";
  }
  else if (num == "5k") {
    fileName += "5k/5000_nodes"; 
  }
  else if (num == "20k") {
    fileName += "20k/20000_nodes";
  }
  else if (num == "50k") {
    fileName += "50k/50000_nodes"; 
  }
  else if (num == "200k") {
    fileName += "200k/200000_nodes";
  }
  else if (num == "1m") {
    fileName += "1m/1000000_nodes";
  }
  else {
    std::cerr << "usage: ./run [LL/LH/HL/HH] [1k/5k/20k/50k/200k/1m]\n";
    std::exit(1);
  }

  std::ifstream file(fileName + ".tsv");
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
  std::ifstream true_file(fileName + "_truePartition.tsv");
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
