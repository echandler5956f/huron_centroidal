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
            casadi::SX prev_final_state;
            casadi::SX curr_initial_state;
            std::vector<double> equality_back(this->state_indices.ndx, 0.0);
            int i = 0;
            for (auto phase : contacts.phase_sequence_)
            {
                auto ps = PseudospectralSegment(d, phase.knot_points, phase.time_value, &this->state_indices, this->Fint);
                /*TODO: Fill with user defined functions, and handle global/phase-dependent/time-varying constraints*/
                std::vector<std::shared_ptr<ConstraintData>> G;
                ps.initialize_expression_graph(this->F, this->L, G);
                this->trajectory.push_back(ps);
                ps.evaluate_expression_graph(J, g);
                ps.fill_ub(lb);
                ps.fill_ub(ub);
                ps.fill_w(w);
                if (i > 0)
                {
                    curr_initial_state = ps.get_initial_state();
                    /*For general jump map functions you can use the following syntax:*/
                    // g.push_back(jump_map_function(casadi::SXVector{prev_final_state, curr_initial_state}).at(0));
                    g.push_back(prev_final_state - curr_initial_state);
                    lb.insert(lb.end(), equality_back.begin(), equality_back.end());
                    ub.insert(ub.end(), equality_back.begin(), equality_back.end());
                }
                prev_final_state = ps.get_final_state();
                if (i == contacts.phase_sequence_.size() - 1)
                {
                    /*Add terminal cost*/
                    J += this->Phi(casadi::SXVector{prev_final_state}).at(0);
                }
                ++i;
            }
        }
    }
}