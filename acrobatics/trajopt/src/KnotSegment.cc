#include "../include/KnotSegment.h"

namespace acrobatics
{

    void KnotSegment::initializeLocalDynamics()
    {
        // f_local = dt * f_global(Xc, mask * Uc);
    }

    void KnotSegment::initializeConstraints()
    {

        // For collocation point "i"

        // Collocation constraint
        // Function(f_local(Xc[i], evaluate_u(i)) - D * Xc[i])

        // Contact constraint
        // vertcat(
        //  vertcat(
        //      ( z_position(Xc[i], foot_index) - contact.desired_height(foot_index) )
        //      ( contact.A(foot_index) * xy_position(Xc[i], foot_index) - contact.b(foot_index) )
        //  ) for each foot in contact
        //)

        // Friction cone constraint
        // vertcat(
        //  (GRF(evaluate_U[i], foot_index ) in friction cone) for each foot in contact
        //)

        // Velocity constraint
        // vertcat(
        //  ( z_velocity(Xc[i]) - (kp * (z_height - contact.z_height_desired(foot_index) )) - kd * contact.z_vel_desired(foot_index) )
        //        for each foot
        //)

        // How do you manage the foot height trajectory?
    }

    void KnotSegment::initializeCollocationMatrices(int num_collocation_points_x);
}