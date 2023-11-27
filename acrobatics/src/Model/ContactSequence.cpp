#include "ContactSequence.h"

namespace acro
{
    namespace contact
    {

        // Makes the combination valid. If an EE is not in contact, it makes the corresponding contact surface NO_SURFACE
        void ContactMode::MakeValid(ContactModeValidity &validity)
        {
            if (combination_definition.size() != contact_surfaces.size())
            {
                validity = ContactModeValidity::DIFFERING_SIZES;
                return;
            }

            for (int i = 0; i < combination_definition.size(); i++)
            {
                bool is_in_contact = combination_definition[i].second;
                if (!is_in_contact)
                {
                    // If it is not in contact, make the contact surface NO_SURFACE
                    contact_surfaces[i] = environment::NO_SURFACE;
                }

                bool surface_is_not_defined = (contact_surfaces[i] == environment::NO_SURFACE);

                if (is_in_contact && surface_is_not_defined)
                {
                    validity = ContactModeValidity::SURFACE_NOT_DEFINED;
                    return;
                }
            }
        }

        const ContactSequence::Phase &ContactSequence::getPhase(int index) { return phase_sequence_[index]; }

        int ContactSequence::addPhase(const ContactMode &mode, int knot_points, double dt)
        {
            // assert that contacts.size() == num_end_effectors_
            Phase new_phase;
            new_phase.mode = mode;
            new_phase.mode.MakeValid();
            new_phase.knot_points = knot_points;
            new_phase.time_value = dt;

            phase_sequence.push_back(new_phase);
            phase_t0_offset_.push_back(dt_);
            dt_ += dt;

            phase_knot0_idx_.push_back(total_knots);
            total_knots_ += knot_points;
        }

        int ContactSequence::getPhaseIndexAtTime(double t, CONTACT_SEQUENCE_ERROR &error_status)
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

        int ContactSequence::getPhaseIndexAtKnot(int knot_idx, CONTACT_SEQUENCE_ERROR &error_status)
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

        void ContactSequence::getPhaseAtTime(double t, Phase &phase, CONTACT_SEQUENCE_ERROR &error_status)
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
    }
}