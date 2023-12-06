#pragma once

#include "Model/ContactSequence.h"
#include "Variables/PseudospectralSegment.h"

namespace acro
{
    namespace variables
    {
        /**
         * @brief The trajectory optimization class
         *
         */
        class TrajectoryOpt
        {
        public:
            /**
             * @brief Construct a new Trajectory Opt object
             *
             */
            TrajectoryOpt();

            /**
             * @brief Initialize the finite elements
             *
             * @param contacts The phase sequence
             * @param d The degree of the finite element polynomials
             */
            void init_finite_elements(contact::ContactSequence contacts, int d);

            /**
             * @brief Optimize and return the solution
             * 
             * @return casadi::DMDict 
             */
            casadi::DMDict optimize();

        private:
            /**
             * @brief A Trajectory is made up of pseudospectral finite elements
             *
             */
            std::vector<PseudospectralSegment> trajectory;

            /**
             * @brief Continuous-time function. The decision variables are infinitesimal deviations from the initial state,
                allowing for states to lie on a manifold. Fint is the function which maps these
                deviations back to the actual state space
             *
             */
            casadi::Function Fint;

            /**
             * @brief Continuous-time function. This function stores the system dynamics
             *
             */
            casadi::Function F;

            /**
             * @brief The "running" or integrated cost function
             *
             */
            casadi::Function L;

            /**
             * @brief The terminal cost function
             *
             */
            casadi::Function Phi;

            /**
             * @brief Casadi solver options
             *
             */
            casadi::Dict opts;

            /**
             * @brief Nonlinear function solver
             *
             */
            casadi::Function solver;

            /**
             * @brief Slicer to get the states
             *
             */
            States state_indices;

            /**
             * @brief Fixed time horizon of the entire trajectory
             *
             */
            double T;

            /**
             * @brief Vector of all decision variables
             * 
             */
            casadi::SXVector w;

            /**
             * @brief Vector of all constraint expressions
             * 
             */
            casadi::SXVector g;

            /**
             * @brief Vector of all constraint lower bounds
             * 
             */
            std::vector<double> lb;


            /**
             * @brief Vector of all constraint upper bounds
             * 
             */
            std::vector<double> ub;

            /**
             * @brief Expression for objective cost
             * 
             */
            casadi::SX J;

            /**
             * @brief Vector of all times where decision variables are evaluated
             * 
             */
            std::vector<double> all_times;
        };
    }
}