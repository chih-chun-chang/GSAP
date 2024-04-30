#include <iostream>
#include "./include/graph.hpp"
#include "./include/partition.hpp"
#include "./include/load.hpp"
#include "./include/eval.hpp"

int main (int argc, char *argv[]) {

  if (argc != 2) {
    std::cerr << "usage: ./run [Number of ints]\n";
    std::exit(1);
  }
  
  int numNodes = std::stoi(argv[1]);

  //auto g = readFile<int, int, unsigned long long>(numNodes);
  auto g = readFile<int, int, unsigned int>(numNodes);
  std::cout << "Number of nodes: " << g.N << std::endl;
  std::cout << "Number of edges: " << g.E << std::endl;

  auto begin = std::chrono::steady_clock::now();
  partition(g);
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  std::cout << "Partitioning time: " <<
    std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
    << " ms" << std::endl;

  eval(g);

}
