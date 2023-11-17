#include <Eigen/Core>
#include <Eigen/Geometry>

namespace acro
{
    namespace environment
    {
        struct SurfaceData
        {
            // Assume it is Z aligned, so, all we need is the height.
            double origin_z_offset;

            // The A and b values defining the surface 2d polytope in the ground frame.
            Eigen::MatrixXd A;
            // A must be an "a by 2" vector.
            Eigen::VectorXd b;

            // The chebychev center of A and b
            Eigen::Vector2d polytope_local_chebyshev_center;
        };

        // The chebyshev center of A and b in 3d. WILL NOT WORK IF THE LOCAL CENTER HAS NOT BEEN COMPUTED.
        Eigen::Vector3d getChebyshevCenter(const SurfaceData &surface_data)
        {
            // returns the chebychev center in the global frame.
            // (Adds a height component if the surface is gravity aligned and in the global frame)

            Eigen::Vector3d c_center;
            c_center.head(2) = surface_data.polytope_local_chebyshev_center;
            c_center.tail(1) = surface_data.origin_z_offset;
        }

        Eigen::VectorXd CalculateChebyshevCenter(const Eigen::MatrixXd &A, const Eigen::VectorXd &b)
        {
            // min (-r)
            // r (scalar)
            // d in Rn
            //  s.t
            //       (r * a_i / ||a_i||) + d  \leq  ||a_i|| * b
            // where a_i = A[i,:] .transpose()

            // or

            // min [[-1] [0 0 ... 0]] * x
            // x = [r;d]
            //  s.t [normalized(a_i) I_(nxn)] * r \leq  b~
            // where b~_i = b_i * ||a_i||

            // this is a linear program.
        }

        class EnvironmentSurfaces
        {
            // EMPTY FOR NOW; DISCUSS WITH YIFU & LEHONG
            // Get k-closest regions to current; convex program.
            // need a way to define surface and surface identifiers
            std::vector<SurfaceIdentifiers> LeggedBody::getKClosestRegions(Eigen::Vector3d ee_pos, int k);
        };
    }
}