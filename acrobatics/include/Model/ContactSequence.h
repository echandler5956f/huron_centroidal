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

            int addPhase(const ContactCombination &contacts, int knot_points, double dt);

            int getPhaseIndexAtTime(double t, CONTACT_SEQUENCE_ERROR &error_status);

            int getPhaseIndexAtKnot(int knot_idx, CONTACT_SEQUENCE_ERROR &error_status);

            void getPhaseAtTime(double t, Phase &phase, CONTACT_SEQUENCE_ERROR &error_status);

            void getPhaseAtKnot(int knot_idx, Phase &phase, CONTACT_SEQUENCE_ERROR &error_status);

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