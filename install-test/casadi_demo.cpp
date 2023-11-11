#include <iostream>
#include <fstream>
#include <ctime>
#include <casadi/casadi.hpp>

using namespace casadi;

int main(int argc, char ** argv) {

  auto opti = casadi::Opti();

  auto x = opti.variable();
  auto y = opti.variable();
  auto z = opti.variable();

  opti.minimize(x*x + 100*z*z);
  opti.subject_to(z+(1-x)*(1-x)-y==0);

  opti.solver("ipopt");
  auto sol = opti.solve();
  
  std::cout << sol.value(x) << ":" << sol.value(y) << std::endl;



  return 0;
}