#include <casadi/casadi.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

#include "../../robot/ContactValue.h"

namespace acrobatics
{
    struct CollocationMatrices
    {
        Eigen::MatrixXd D;
        Eigen::MatrixXd B;
        Eigen::MatrixXd C;
    };

    class KnotSegment
    {
        KnotSegment(std::shared_ptr<casadi::MX> f_system, double dt, KnotSegmentData knot_segment_data) : f_global_(f_system),
                                                                                                          h(dt), knot_segment_data_(knot_segment_data)
        {
            initializeLocalDynamics();
            initializeConstraints();
            initializeCollocationMatrices(num_points);
        }

        casadi::MX &getCollocationConstraint() { return collocation_constraint_; }
        casadi::MX &getFrictionConeConstraint() { return friction_cone_constraint_; }
        casadi::MX &getVelocityConstraint() { return velocity_constraint_; }
        casadi::MX &getContactConstraint() { return contact_constraint_; }

        struct KnotSegmentData
        {
            // for creating the mask
            int num_limbs;

            int num_collocation_points_x;
            int num_collocation_points_u;

            // number of states of the system
            int num_states;
            // number of control inputs in the system
            int num_control_inputs;

            ContactValue contact_value;
        };

    private:
        void initializeLocalDynamics();
        void initializeConstraints();
        void initializeCollocationMatrices(int num_collocation_points_x);

        casadi::MX collocation_constraint_;
        casadi::MX friction_cone_constraint_;
        casadi::MX velocity_constraint_;
        casadi::MX contact_constraint_;

        std::shared_ptr<casadi::MX> f_global_;
        // normalized for timing difference and masking contact legs
        casadi::MX f_local_;
        double h_;

        KnotSegmentData knot_segment_data_;
        // Vectors giving the collocation values(which are vectors in themselves). Is it a vector of SX?
        casadi::SX Uc_;
        casadi::SX Xc_;

        // Vectors signifying the initial and final states.
        casadi::SX X0_;
        casadi::SX Xf_;

        // Should these be calculated a-priori and passed in, or created on construction?
        CollocationMatrices X_collocation_;
        CollocationMatrices U_collocation_;
    };
}