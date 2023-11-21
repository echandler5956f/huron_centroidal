#pragma once

#include "Variables/PseudospectralSegment.h"

namespace acro
{
    namespace variables
    {
        class Trajectory
        {
        public:
            Trajectory();

        private:
            void init_finite_elements();

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

            /*Slicer to get the states*/
            States state_indices;

            /*Fixed time horizon*/
            double T;
        };
    }
}