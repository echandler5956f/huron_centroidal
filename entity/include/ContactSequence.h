#include "EndEffector.h"
namespace acrobatics
{
    namespace contact
    {
        class ExtendContactSequenceRule
        {
            ExtendContactSequenceRule() {}
        };

        class ContactSequence
        {
        public:
            enum CONTACT_SEQUENCE_ERROR
            {
                OK,
                NOT_IN_DT
            };

            ContactSequence(int num_end_effectors) : num_end_effectors_(num_end_effectors) {}

            // copy, move, etc constructors
            // destructor
            // SPECIAL COPY CONSTRUCTOR, FOR MPC CONTEXT
            ContactSequence(const ContactSequence &cs, double time_offset, ExtendContactSequenceRule extend_contact_sequence_rule)
            {
                CONTACT_SEQUENCE_ERROR construction_status = CONTACT_SEQUENCE_ERROR::OK;
                int current_phase_index = getPhaseIndexAtTime(time_offset, construction_status);
                int current_knot_index = getKnotAtTime(time_offset, construction_status);
                // assert(construction_status == OK)

                // all previous indices need to be removed, and the knot points of the current phase need to be updated.
                // we need to figure out how to preserve computations of dynamics and the sort. That's a later problem.

                // We also need to backfill the last "time_offset" seconds using some contact sequence extension rule. Unimplemented.

                // does this computation happen here, or is there an MPC controller class? If there is an mpc controller class,
                //  there needs to be a good way to abstract away the choosing of initial guesses, setting up of dynamics & casadi problem.

                // Maybe this is not it.
            }

            // Does the phase timing change? if so, then the _t0_offset and dt_ need to change.
            struct Phase
            {
                ContactCombination contacts;
                int knot_points = 1;
                double time_value = 1;
            };

            int addPhase(const ContactCombination &contacts, int knot_points, double dt)
            {
                // assert that contacts.size() == num_end_effectors_
                Phase new_phase;
                new_phase.contacts = contacts;
                new_phase.knot_points = knot_points;
                new_phase.time_value = dt;

                phase_sequence.push_back(new_phase);
                phase_t0_offset_.push_back(dt_);
                dt_ += dt;

                phase_knot0_idx_.push_back(total_knots);
                total_knots_ += knot_points;
            }

            int getPhaseIndexAtTime(double t, CONTACT_SEQUENCE_ERROR &error_status)
            {
                if ((t < 0) || (t > dt_))
                {
                    error_status = CONTACT_SEQUENCE_ERROR::NOT_IN_DT;
                    return -1;
                }

                for (int i = num_phases() - 1; i > 0; i--)
                {
                    bool is_in_phase_i = (t >= phase_t0_offset_[i]);
                    if (is_in_phase_i)
                    {
                        error_status = CONTACT_SEQUENCE_ERROR::OK;
                        return i;
                    }
                }
            }

            int getPhaseIndexAtKnot(int knot_idx, CONTACT_SEQUENCE_ERROR &error_status)
            {
                if ((knot_idx < 0) || (knot_idx >= total_knots_))
                {
                    error_status = CONTACT_SEQUENCE_ERROR::NOT_IN_DT;
                    return -1;
                }

                for (int i = num_phases() - 1; i > 0; i--)
                {
                    bool is_in_phase_i = (knot_idx >= phase_knot0_idx_[i]);
                    if (is_in_phase_i)
                    {
                        error_status = CONTACT_SEQUENCE_ERROR::OK;
                        return i;
                    }
                }
            }

            int getKnotAtTime(double t, CONTACT_SEQUENCE_ERROR &error_status)
            {
                // not sure how the knot points are implemented.
            }

            void getPhaseAtTime(double t, Phase &phase, CONTACT_SEQUENCE_ERROR &error_status)
            {
                int phase_index = getPhaseIndexAtTime(t, error_status);
                if (error_status != CONTACT_SEQUENCE_ERROR::OK)
                {
                    return;
                }

                phase = phase_sequence[phase_index];
            }

            void getPhaseAtKnot(int knot_idx, Phase &phase, CONTACT_SEQUENCE_ERROR &error_status)
            {
                int phase_index = getPhaseIndexAtKnot(knot_idx, error_status);
                if (error_status != CONTACT_SEQUENCE_ERROR::OK)
                {
                    return;
                }

                phase = phase_sequence[phase_index];
            }

            int num_phases() { return phase_sequence.size(); }

        private:
            std::vector<Phase> phase_sequence_;
            std::vector<double> phase_t0_offset_;
            std::vector<int> phase_knot0_idx_;
            double dt_ = 0;
            int total_knots_ = 0;
            int num_end_effectors_;
        };
    }
}