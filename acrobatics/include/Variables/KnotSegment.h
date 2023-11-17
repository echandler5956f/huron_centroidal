#include <pinocchio/autodiff/casadi.hpp>
#include <vector>

namespace acro
{
    namespace variables
    {
        class KnotSegment
        {
            public:
                KnotSegment();
            private:
                // Actual decision variables
                std::vector<casadi::SX> Xc_var;
                std::vector<casadi::SX> U_var;
                casadi::SX X0_var;
                casadi::SX Xf_expr;
        };
    }
}