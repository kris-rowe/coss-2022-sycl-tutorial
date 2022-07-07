#ifndef _STATS_HPP_
#define _STATS_HPP_

#include <iostream>
#include <numeric>
#include <vector>

namespace stats
{
  template<typename T>
  struct stats_t
  {
    T total;
    T mean;
    T min;
    T max;
    T median;
    T variance;
    T stddev;
    std::string units;
  };

  template<typename T>
  inline stats_t<T> computeStats(std::vector<T>& dataset,std::string units)
  {
    stats_t<T> s;
    s.units = units;
    std::sort(dataset.begin(), dataset.end());
    s.total = std::accumulate(dataset.begin(), dataset.end(), T(0));
    s.mean = s.total / static_cast<T>(dataset.size());
    s.min = dataset.front();
    s.median = dataset[dataset.size()/2L];
    s.max = dataset.back();

    s.variance = T(0);
    for (auto &x : dataset)
    {
      s.variance += (x * x);
    }
    s.variance /= static_cast<T>(dataset.size());
    s.variance -= (s.mean * s.mean);
    s.stddev = std::sqrt(s.variance);
    return s;
  }

  template<typename T,size_t N=6>
  void printStats(const stats_t<T>& stats) {
    std::cout.precision(N);
    std::cout << "mean: " << std::scientific << stats.mean << stats.units << "\n";
    std::cout << "std: " << std::scientific << stats.mean << stats.units << "\n";
    std::cout << "min: " << std::scientific << stats.min << stats.units << "\n";
    std::cout << "max: " << std::scientific <<  stats.max << stats.units << "\n";
    std::cout << "\n";
  }

}
#endif
