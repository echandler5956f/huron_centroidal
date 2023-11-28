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
            for (auto phase : contacts.phase_sequence_)
            {
                auto ps = PseudospectralSegment(d, phase.knot_points, phase.time_value, &this->state_indices, this->Fint);
                /*TODO: Fill with user defined functions, and handle global/phase-dependent/time-varying constraints*/
                std::vector<casadi::Function> G;
                ps.initialize_expression_graph(this->F, this->L, G);
                this->trajectory.push_back(ps);
            }
        }
    }
}