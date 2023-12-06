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
            this->w.clear();
            this->g.clear();
            this->lb.clear();
            this->ub.clear();
            this->J = 0;

            this->all_times.clear();

            casadi::SX prev_final_state;
            casadi::SX curr_initial_state;

            std::vector<double> equality_back(this->state_indices.ndx, 0.0);
            std::size_t i = 0;
            for (auto phase : contacts.phase_sequence_)
            {
                auto ps = PseudospectralSegment(d, phase.knot_points, phase.time_value, &this->state_indices, this->Fint);
                /*TODO: Fill with user defined functions, and handle global/phase-dependent/time-varying constraints*/
                std::vector<std::shared_ptr<ConstraintData>> G;
                ps.initialize_expression_graph(this->F, this->L, G);
                this->trajectory.push_back(ps);
                ps.evaluate_expression_graph(this->J, this->g);
                ps.fill_lb_ub(this->lb, this->ub);
                ps.fill_w(this->w);
                ps.fill_times(this->all_times);
                if (i > 0)
                {
                    curr_initial_state = ps.get_initial_state();
                    /*For general jump map functions you can use the following syntax:*/
                    // g.push_back(jump_map_function(casadi::SXVector{prev_final_state, curr_initial_state}).at(0));
                    this->g.push_back(prev_final_state - curr_initial_state);
                    this->lb.insert(this->lb.end(), equality_back.begin(), equality_back.end());
                    this->ub.insert(this->ub.end(), equality_back.begin(), equality_back.end());
                }
                prev_final_state = ps.get_final_state();
                if (i == contacts.phase_sequence_.size() - 1)
                {
                    /*Add terminal cost*/
                    this->J += this->Phi(casadi::SXVector{prev_final_state}).at(0);
                }
                ++i;
            }
        }

        casadi::DMDict TrajectoryOpt::optimize()
        {
            casadi::SXDict nlp = {{"x", vertcat(this->w)},
                                  {"f", this->J},
                                  {"g", vertcat(this->g)}};
            this->solver = casadi::nlpsol("solver", "ipopt", nlp, this->opts);
            casadi::DMDict arg;
            arg["lbg"] = this->lb;
            arg["ubg"] = this->ub;
            return this->solver(arg);
        }
    }
}