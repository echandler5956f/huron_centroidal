#include "Model/ContactSequence.h"

acro::contact::ContactSequence::ContactSequence(int num_end_effectors)
{
    this->num_end_effectors_ = num_end_effectors;
}

int acro::contact::ContactSequence::addPhase(const ContactCombination &contacts, int knot_points, double dt)
{
    // assert that contacts.size() == num_end_effectors_
    Phase new_phase;
    new_phase.contacts = contacts;
    new_phase.knot_points = knot_points;
    new_phase.time_value = dt;

    this->phase_sequence_.push_back(new_phase);
    phase_t0_offset_.push_back(dt_);
    dt_ += dt;

    phase_knot0_idx_.push_back(this->total_knots_);
    total_knots_ += knot_points;
}

int acro::contact::ContactSequence::getPhaseIndexAtTime(double t, CONTACT_SEQUENCE_ERROR &error_status)
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

int acro::contact::ContactSequence::getPhaseIndexAtKnot(int knot_idx, CONTACT_SEQUENCE_ERROR &error_status)
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

int acro::contact::ContactSequence::getKnotAtTime(double t, CONTACT_SEQUENCE_ERROR &error_status)
{
    // not sure how the knot points are implemented.
}

void acro::contact::ContactSequence::getPhaseAtTime(double t, Phase &phase, CONTACT_SEQUENCE_ERROR &error_status)
{
    int phase_index = getPhaseIndexAtTime(t, error_status);
    if (error_status != CONTACT_SEQUENCE_ERROR::OK)
    {
        return;
    }

    phase = this->phase_sequence_[phase_index];
}

void acro::contact::ContactSequence::getPhaseAtKnot(int knot_idx, Phase &phase, CONTACT_SEQUENCE_ERROR &error_status)
{
    int phase_index = getPhaseIndexAtKnot(knot_idx, error_status);
    if (error_status != CONTACT_SEQUENCE_ERROR::OK)
    {
        return;
    }

    phase = this->phase_sequence_[phase_index];
}

int acro::contact::ContactSequence::num_phases()
{
    return this->phase_sequence_.size();
}