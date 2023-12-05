#include "Variables/TrajectoryOpt.h"

namespace acro
{
    namespace variables
    {
        TrajectoryOpt::TrajectoryOpt()
        {
        }

        void TrajectoryOpt::init_finite_elements(contact::ContactSequence contacts, int d)
        {
            casadi::SXVector w;
            casadi::SXVector g;
            std::vector<double> lb;
            std::vector<double> ub;
            casadi::SX J = 0;
            for (auto phase : contacts.phase_sequence_)
            {
                auto ps = PseudospectralSegment(d, phase.knot_points, phase.time_value, &this->state_indices, this->Fint);
                /*TODO: Fill with user defined functions, and handle global/phase-dependent/time-varying constraints*/
                std::vector<std::shared_ptr<ConstraintData>> G;
                ps.initialize_expression_graph(this->F, this->L, G);
                this->trajectory.push_back(ps);

                // This is pretty bad in terms of memory. pls fix and use pointers
                casadi::SXVector result = ps.evaluate_expression_graph(J);
                auto ps_lb = ps.get_lb();
                auto ps_ub = ps.get_ub();
                lb.insert(lb.end(), ps_lb.begin(), ps_lb.end());
                ub.insert(ub.end(), ps_ub.begin(), ps_ub.end());
                g.insert(g.end(), result.begin(), result.end());
            }
        }
    }
}