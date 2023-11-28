#include "Variables/Constraint.h"

namespace acro
{
    namespace variables
    {

        casadi::Function LinearStateInputConstraint::getLinearApproximation(const casadi::Function &F, const casadi::SX &x, const casadi::SX &u, const casadi::SX &x0, const casadi::SX &u0) const
        {
            // return casadi::Function("Linearized Function", std::vector<casadi::SX>{x, u, x0, u0}, std::vector<casadi::SX>{rhs});
            return F;
        }
    }
}