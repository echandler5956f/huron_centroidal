#include "Model/EndEffector.h"

namespace acro
{
    namespace contact
    {

        class ContactSequence
        {
        public:
            enum CONTACT_SEQUENCE_ERROR
            {
                OK,
                NOT_IN_DT
            };

            ContactSequence(int num_end_effectors) : num_end_effectors_(num_end_effectors) {}

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

            // we will fill this out as needed.
        private:
            std::vector<Phase> phase_sequence_;
            std::vector<double> phase_t0_offset_;
            std::vector<int> phase_knot0_idx_;
            double dt_ = 0;
            int total_knots_ = 0;
            int num_end_effectors_;
            int num_end_effectors;
        };
    }
}