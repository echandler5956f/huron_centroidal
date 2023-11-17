#include "Variables/KnotSegment.h"

namespace acro
{
    namespace variables
    {
        class PseudospectralSegment
        {
            public:
                PseudospectralSegment();
            private:
                /*A pseudospectral finite element is made up of knot segments*/
                std::vector<KnotSegment> traj_segment;

                /* 
                Implicit discrete-time functions:
                    collocation_constraint_map: This function map returns the vector of collocation equations 
                    necessary to match the derivative defect between the approximated dynamics and actual system 
                    dynamics.

                    xf_constraint_map: The map which matches the approximated final state expression with the initial 
                    state of the next segment

                    q_cost_fold: The accumulated cost across all the knot segments found using quadrature rules.
                */
                casadi::Function collocation_constraint_map;
                casadi::Function xf_constraint_map;
                casadi::Function q_cost_fold;

                // Variables used to build the expression graphs
                std::vector<casadi::SX> Xc;
                std::vector<casadi::SX> U;
                casadi::SX X0;
                casadi::SX Lc;
        };
    }
}