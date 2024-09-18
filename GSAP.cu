#include <iostream>
#include "./GSAP/graph.hpp"
#include "./GSAP/partition.hpp"
#include "./GSAP/load.hpp"
#include "./GSAP/evaluate/eval.hpp"

int main (int argc, char *argv[]) {

  if (argc != 3) {
    std::cerr << "usage: ./run [LL/LH/HL/HH] [1k/5k/20k/50k/200k/1m]\n";
    std::exit(1);
  }
  
  //int numNodes = std::stoi(argv[1]);

  std::string arg1 = argv[1];
  std::string arg2 = argv[2];

  //auto g = readFile<int, int, unsigned long long>(numNodes);
  auto g = readFile<int, int, unsigned int>(arg1, arg2);
  std::cout << "Number of Vertices                : " << g.N << std::endl;
  std::cout << "Number of Edges                   : " << g.E << std::endl;

  auto begin = std::chrono::steady_clock::now();
  partition(g);
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  std::cout << "Partitioning Runtime (ms)         : " <<
    std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
    << std::endl;

  eval(g);

}
