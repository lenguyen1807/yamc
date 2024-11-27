#include "lib.hpp"

auto main() -> int
{
  auto const lib = library {};

  return lib.name == "mnist-cpp" ? 0 : 1;
}
