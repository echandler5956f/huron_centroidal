#pragma once

#include <vector>
#include <string>
#include "Variables/Constraint.h"
#include "Variables/Objective.h"
#include <cassert>
#include "Model/LeggedBody.h"

namespace acro
{
    namespace variables
    {
        struct KnotSegment
        {
            /*Actual decision variables*/
            std::vector<casadi::SX> dXc_var;
            std::vector<casadi::SX> U_var;
            casadi::SX dX0_var;
        };

        class LagrangePolynomial
        {
        public:
            /*Default constructor*/
            LagrangePolynomial();

            /*Compute and store the coefficients for a given degree and collocation scheme*/
            void compute_matrices(int d_, const std::string &scheme = "radau");

            /*Perform symbolic Lagrange Interpolation, which, given a time from the Lagrange time scale, interpolates terms to find the value at time t.*/
            casadi::SX lagrange_interpolation(double t, std::vector<casadi::SX> terms);

            /*Degree of the polynomial*/
            int d;
            /*The roots of the polynomial*/
            Eigen::VectorXd tau_root;
            /*Quadrature coefficients*/
            Eigen::VectorXd B;
            /*Collocation coeffficients*/
            Eigen::MatrixXd C;
            /*Continuity coefficients*/
            Eigen::VectorXd D;
        };

        class PseudospectralSegment
        {
        public:
            /*Constructor takes a polynomial degree d, a number of knot segments knot_num_, a period of each knot segment h_, a pointer to the state index helper state_indices_, integrator function*/
            PseudospectralSegment(int d, int knot_num_, double h_, States *state_indices_, casadi::Function Fint_);

            /*Initialize the relevant expressions*/
            void initialize_expression_variables(int d);

            /*Create all the knot segments*/
            void initialize_knot_segments();

            /*Build the function graph*/
            void initialize_expression_graph(casadi::Function F_, casadi::Function L_x, casadi::Function L_u);

        private:
            /*A pseudospectral finite element is made up of knot segments*/
            std::vector<KnotSegment> traj_segment;

            /*
            Implicit discrete-time functions:
                collocation_constraint_map: This function map returns the vector of collocation equations
                necessary to match the derivative defect between the approximated dynamics and actual system
                dynamics.

                xf_constraint_map: The map which matches the approximated final state expression with the initial
                state of the next segment

                q_cost_fold: The accumulated cost across all the knot segments found using quadrature rules.
            */
            casadi::Function collocation_constraint_map;
            casadi::Function xf_constraint_map;
            casadi::Function q_cost_fold;

            casadi::Function Fint;

            /*Variables used to build the expression graphs*/
            std::vector<casadi::SX> dXc;
            std::vector<casadi::SX> Uc;
            casadi::SX dX0;
            casadi::SX Lc;

            /*Helper class to store polynomial information*/
            LagrangePolynomial U_poly;
            LagrangePolynomial dX_poly;

            /*Helper class for indexing the state variables*/
            States *st_m;

            /*Number of knot segments*/
            int knot_num;
            /*Period of EACH KNOT SEGMENT within this pseudospectral segment*/
            double h;
        };
    }
}