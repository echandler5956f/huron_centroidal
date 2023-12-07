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

        using tuple_size_t = std::tuple<std::size_t, std::size_t>;

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
            const casadi::SX lagrange_interpolation(double t, const casadi::SXVector terms);

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
             * @brief Fill all times with the time vector from this segment
             * 
             * @param all_times 
             */
            void fill_times(std::vector<double> &all_times);

            /**
             * @brief Create all the knot segments
             * @param Starting state to integrate from. Can be a constant
             *
             */
            void initialize_knot_segments(casadi::SX x0);

            /**
             * @brief Build the function graph
             *
             * @param F Function for the system dynamics
             * @param L Integrated cost
             * @param G Vector of constraint data
             */
            void initialize_expression_graph(casadi::Function &F, casadi::Function &L, std::vector<std::shared_ptr<ConstraintData>> G);

            /**
             * @brief Evaluate the expressions with the actual decision variables
             *
             * @param J0 Accumulated cost so far
             * @param g Constraint vector to fill
             */
            void evaluate_expression_graph(casadi::SX &J0, casadi::SXVector &g);

            /**
             * @brief Get the initial state deviant
             *
             * @return casadi::SX 
             */
            casadi::SX get_initial_state_deviant();

            /**
             * @brief Get the final state deviant
             *
             * @return casadi::SX 
             */
            casadi::SX get_final_state_deviant();

            /**
             * @brief Get the actual final state
             * 
             * @return casadi::SX 
            */
            casadi::SX get_final_state();

            /**
             * @brief Get lb/ub and fill
             *
             * @param lb To fill
             * @param ub To fill
             */
            void fill_lb_ub(std::vector<double> &lb, std::vector<double> &ub);

            /**
             * @brief Get w and fill
             *
             * @param w To fill
             */
            void fill_w(casadi::SXVector &w);

            /**
             * @brief Returns the starting and ending index in w (call after fill_w!)
             * 
             * @return tuple_size_t 
             */
            tuple_size_t get_range_idx_decision_variables();

            /**
             * @brief Returns the starting and ending index in g (call after evaluate_expression_graph!)
             * 
             * @return tuple_size_t
             */
            tuple_size_t get_range_idx_constraint_expressions();

            /**
             * @brief Returns the starting and ending index in g (call after fill_lb_ub!). This should match get_range_idx_constraint_expressions
             * 
             * @return tuple_size_t 
             */
            tuple_size_t get_range_idx_bounds();

        private:
            /**
             * @brief Collocation state decision variables
             *
             */
            casadi::SXVector dXc_var_vec;

            /**
             * @brief Collocation input decision variables
             *
             */
            casadi::SXVector U_var_vec;

            /**
             * @brief Collocation input decision expressions at the state collocation points 
             * (decision variables of control and state are potentially approximated by different degree polynomials)
             *
             */
            casadi::SXVector U_at_c_vec;

            /**
             * @brief Knot point deviants state decision variables
             *
             */
            casadi::SXVector dX0_var_vec;

            /**
             * @brief Knot point state expressions (integral functions of the deviants)
             *
             */
            casadi::SXVector X0_var_vec;

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
            casadi::SXVector dXc;

            /**
             * @brief Collocation inputs used to build the expression graphs
             *
             */
            casadi::SXVector Uc;

            /**
             * @brief Knot states deviants used to build the expression graphs
             *
             */
            casadi::SX dX0;

            /**
             * @brief Knot states used to build the expression graphs
             *
             */
            casadi::SX X0;

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

            /**
             * @brief Starting and ending index of the decision variables in w corresponding to this segment
             * 
             */
            tuple_size_t w_range;

            /**
             * @brief Starting and ending index of the constraint expressions in g corresponding to this segment. This should match lb_ub_range
             * 
             */
            tuple_size_t g_range;

            /**
             * @brief Starting and ending index of the bounds in lb/ub corresponding to this segment. This should match g_range
             * 
             */
            tuple_size_t lb_ub_range;
        };
    }
}