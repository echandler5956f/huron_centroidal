#include "TrajectoryGeneration.h"

namespace variables
{

    template <class Sym>
    bool ProblemSetup<Sym>::CheckValidity()
    {
        return contact_sequence.getPhaseAtKnot(0).mode == init_condition.init_mode;
    }
}
