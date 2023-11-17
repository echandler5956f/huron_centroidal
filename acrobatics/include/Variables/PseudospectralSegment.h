#include < pinocchio / autodiff / casadi.hpp>
#include <vector>

#include "../Model/LeggedBody.h"

namespace acro
{
    namespace variables
    {
        struct KnotSegment
        {
            // Actual decision variables
            std::vector<casadi::SX> Xc_var;
            std::vector<casadi::SX> U_var;
            casadi::SX X0_var;
            casadi::SX Xf_expr;
        };

        class PseudospectralSegment
        {
        public:
            PseudospectralSegment(contact::ContactCombination contact_combination);

            static void compute_collocation_matrices(int d, Eigen::VectorXd &B, Eigen::MatrixXd &C, Eigen::VectorXd &D, const std::string &scheme = "radau");

        private:
            void InitConstraints();
            void InitMask();

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

            struct KnotSegmentConstraints
            {
                casadi::Function friction_cone_constraint;
                casadi::Function contact_constraint;
                casadi::Function velocity_constraint;
            };

            // PseudospectralSegmentConstraint functions are maps of
            // KnotSegmentConstraint functions. They are still functions,
            // which makes the datatypes the same.
            typedef KnotSegmentConstraints PseudospectralSegmentConstraints;

            // Constriant on Xcs and Us for one knot segment
            KnotSegmentConstraints knot_segment_constraints_;
            // Constraints on the entire pseudospectral segment;
            //  a map of knot segment constraitns
            PseudospectralSegmentConstraints pseudospectral_segment_constraints_;

            // Variables used to build the expression graphs
            std::vector<casadi::SX> Xc;
            std::vector<casadi::SX> U;
            casadi::SX X0;
            casadi::SX Lc;

            contact::ContactCombination contact_combination_;
            Eigen::VectorXd contact_mask_;
        };
    }
}