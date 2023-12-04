#pragma once

#include <vector>
#include <string>
#include <cassert>
#include "Variables/States.h"
#include "Variables/Constraint.h"
#include <Eigen/Sparse>

namespace acro
{
    namespace variables
    {
        /**
         * @brief Knot segments which Pseudospectral Segments are composed o
         *
         */
        struct KnotSegment
        {
            /**
             * @brief Collocation state decision variables
             *
             */
            std::vector<casadi::SX> dXc_var;

            /**
             * @brief Collocation input decision variables
             *
             */
            std::vector<casadi::SX> U_var;

            /**
             * @brief Knot point state decision variables
             *
             */
            casadi::SX dX0_var;
        };

        /**
         * @brief Helper class for storing polynomial information
         *
         */
        class LagrangePolynomial
        {
        public:
            /**
             * @brief Construct a new Lagrange Polynomial object
             *
             */
            LagrangePolynomial(){};

            /**
             * @brief Construct a new Lagrange Polynomial object. Compute and store the coefficients for a given degree and collocation scheme
             *
             * @param d_ Degree of the polynomial
             * @param scheme Collocation scheme: "radau" or "legendre"
             */
            LagrangePolynomial(int d_, const std::string &scheme = "radau");

            /**
             * @brief Perform symbolic Lagrange Interpolation, which, given a time from the Lagrange time scale, interpolates terms to find the value at time t
             *
             * @param t Time to interpolate at
             * @param terms Terms at knot points to use for interpolation
             * @return const casadi::SX Resultant expression for the symbolic interpolated value
             */
            const casadi::SX lagrange_interpolation(double t, const std::vector<casadi::SX> terms);

            /**
             * @brief Degree of the polynomial
             *
             */
            int d;

            /**
             * @brief The roots of the polynomial
             *
             */
            Eigen::VectorXd tau_root;

            /**
             * @brief Quadrature coefficients
             *
             */
            Eigen::VectorXd B;

            /**
             * @brief Collocation coeffficients
             *
             */
            Eigen::MatrixXd C;

            /**
             * @brief Continuity coefficients
             *
             */
            Eigen::VectorXd D;
        };

        /**
         * @brief PseudospectalSegment class
         *
         */
        class PseudospectralSegment
        {
        public:
            /**
             * @brief Construct a new Pseudospectral Segment object
             * 
             */
            PseudospectralSegment(){};

            /**
             * @brief Construct a new Pseudospectral Segment object
             *
             * @param d Polynomial degree
             * @param knot_num_ Number of knots in the segment
             * @param h_ Period of each knot segment
             * @param state_indices_ Pointer to the state indices helper
             * @param Fint_ Integrator function
             */
            PseudospectralSegment(int d, int knot_num_, double h_, States *state_indices_, casadi::Function &Fint_);

            /**
             * @brief Initialize the relevant expressions
             *
             * @param d Polynomial degree
             */
            void initialize_expression_variables(int d);

            /**
             * @brief Initialize the vector of all times which constraints are evaluated at
             * 
             */
            void initialize_time_vector();

            /**
             * @brief Create all the knot segments
             *
             */
            void initialize_knot_segments();

            /**
             * @brief Build the function graph
             *
             * @param F Function for the system dynamics
             * @param L Integrated cost
             * @param G Vector of constraint data
             */
            void initialize_expression_graph(casadi::Function &F, casadi::Function &L, std::vector<std::shared_ptr<ConstraintData>> G);

        private:
            /**
             * @brief A pseudospectral finite element is made up of knot segments
             *
             */
            std::vector<KnotSegment> traj_segment;

            /**
             * @brief Implicit discrete-time function map. This function map returns the vector of collocation equations
                necessary to match the derivative defect between the approximated dynamics and actual system
                dynamics
             *
             */
            casadi::Function collocation_constraint_map;

            /**
             * @brief Implicit discrete-time function map. The map which matches the approximated final state expression with the initial
                state of the next segment
             *
             */
            casadi::Function xf_constraint_map;

            /**
             * @brief Implicit discrete-time function map. The accumulated cost across all the knot segments found using quadrature rules
             *
             */
            casadi::Function q_cost_fold;

            /**
             * @brief User defined constraints, which are functions with certain bounds associated with them
             *
             */
            std::vector<casadi::Function> general_constraint_maps;

            /**
             * @brief Lower bounds associated with the general constraint maps
             *
             */
            casadi::DM general_lb;

            /**
             * @brief Upper bounds associated with the general constraint maps
             *
             */
            casadi::DM general_ub;

            /**
             * @brief Integrator function
             *
             */
            casadi::Function Fint;

            /**
             * @brief Collocation states used to build the expression graphs
             *
             */
            std::vector<casadi::SX> dXc;

            /**
             * @brief Collocation inputs used to build the expression graphs
             *
             */
            std::vector<casadi::SX> Uc;

            /**
             * @brief Knot states used to build the expression graphs
             *
             */
            casadi::SX dX0;

            /**
             * @brief Accumulator expression used to build the expression graphs
             *
             */
            casadi::SX Lc;

            /*Helper class to store polynomial information*/
            /**
             * @brief Input polynomial
             *
             */
            LagrangePolynomial U_poly;

            /**
             * @brief State polynomial
             *
             */
            LagrangePolynomial dX_poly;

            /**
             * @brief Helper for indexing the state variables
             *
             */
            States *st_m;

            /**
             * @brief Number of knot segments
             *
             */
            int knot_num;

            /**
             * @brief Vector of all times
             *
             */
            casadi::DM times;

            /**
             * @brief Period of EACH KNOT SEGMENT within this pseudospectral segment
             *
             */
            double h;

            /**
             * @brief Total period (helper variable calculated from h and knot_num)
             *
             */
            double T;
        };
    }
}