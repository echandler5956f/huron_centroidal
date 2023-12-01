#pragma once

#include "Model/ContactSequence.h"
#include "Variables/PseudospectralSegment.h"

namespace acro
{
    namespace variables
    {
        class TrajectoryOpt
        {
        public:
            TrajectoryOpt();

        private:
            void init_finite_elements(contact::ContactSequence contacts, int d);

            /*A Trajectory is made up of pseudospectral finite elements*/
            std::vector<PseudospectralSegment> trajectory;

            /*
            Continuous-time functions:
                Fint: The decision variables are infinitesimal deviations from the initial state,
                allowing for states to lie on a manifold. Fint is the function which maps these
                deviations back to the actual state space.

                F: The system dynamics.

                L: The 'running', or integrated cost.

                Phi: The terminal cost.
            */
            casadi::Function Fint;
            casadi::Function F;
            casadi::Function L;
            casadi::Function Phi;

            casadi::Dict opts;
            casadi::Function solver;

            /*Slicer to get the states*/
            States state_indices;

            /*Fixed time horizon*/
            double T;
        };
    }
}