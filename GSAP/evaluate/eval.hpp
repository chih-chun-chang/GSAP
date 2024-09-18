#include "../graph.hpp"
#include "evaluate.hpp"

template <typename T, typename V, typename W>
void eval(const Graph<T, V, W>& g) {
  bf::evaluate<V>(g.truePartitions, g.partitions);
}
