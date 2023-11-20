#include "States.h"

namespace variables
{
    template <class Sym>
    struct Target
    {
        Target(Sym set_target_vars, States set_state_def) : target_vars(set_target_vars), state_def(set_state_def)
        {
            int size_of_state = state_def.ndh + state_def.nq + state_def.nvb + state_def.nx;
            Q = Eigen::MatrixXd::Identity(size_of_state, size_of_state);
        }

        Sym target_vars;
        States state_def;
        // Cost matrix
        Eigen::MatrixXd Q;
    };

    template <class Sym>
    struct ProblemSetup
    {

        ProblemSetup(Sym set_x0_vars,
                     States set_state_def,
                     contact::ContactCombination set_init_mode) : x0_vars(set_x0_vars),
                                                                  state_def(set_state_def), init_mode(set_init_mode) {}

        Sym x0_vars;
        States state_def;
        contact::ContactCombination init_mode
    };
}