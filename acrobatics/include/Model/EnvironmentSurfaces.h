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

        template <class T>
        void PointViolation(const SurfaceData &region, const Eigen::Vector2<T> &point, Eigen::VectorX<T> &ineq_violation)
        {
            ineq_violation = region.A * point - region.b;
        }
        template <class T>
        void PointViolation(const SurfaceData &region, const Eigen::Vector3<T> &point, Eigen::VectorX<T> &ineq_violation, Eigen::VectorX<T> &eq_violation)
        {
            ineq_violation = region.A * point.head(2) - region.b;
            eq_violation = point.tail(1) - region.origin_z_offset;
        }

        bool isInRegion(const SurfaceData &region, const Eigen::Vector2d &point)
        {
            Eigen::VectorXd violation;
            PointViolation(region, point, violation);
            return (violation.array() <= Eigen::VectorXd::Zeros(violation.size()).array()).all()
        }

        bool isOnRegion(const SurfaceData &region, const Eigen::Vector3d &point)
        {
            Eigen::VectorXd ineq_violation;
            Eigen::Vector1d eq_violation;
            PointViolation(region, point, violation, eq_violation);
            bool ineq_satisfied = (ineq_violation.array() <= Eigen::VectorXd::Zeros(ineq_violation.size()).array()).all();
            bool eq_satisfied = (eq_violation[0] <= 1e-6) && (eq_violation[0] >= -1e-6);
            return ineq_satisfied && eq_satisfied;
        }

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

        class EnvironmentSurfaces : public std::vector<SurfaceData>
        {
            EnvironmentSurfaces() : std::vector<SurfaceData> {}

            std::vector<int> getSurfacesUnder(const Eigen::Vector2d &ee_pos)
            {
                std::vector<int> surface_indeces;
                for (int i = 0; i < this->size(); i++)
                {
                    bool is_in_region = isInRegion((*this)[i], ee_pos);
                    if (is_in_region)
                    {
                        surface_indeces.push_back(i);
                    }
                }
                return surface_indeces;
            }

            std::vector<SurfaceData> getSurfacesFromIndeces(const std::vector<int> indeces)
            {

                std::vector<SurfaceData> surfaces;
                for (int i = 0; i < indeces.size(); i++)
                {
                    surfaces.push_back((*this)[indeces[i]]);
                }
                return surfaces;
            }

            // Generate the straight shot trajectory of each limb from the starting to the target
            // and sample to find surfaces underneath

            // Get k-closest regions to current; convex program.
            std::vector<int>
            LeggedBody::getKClosestRegions(Eigen::Vector3d ee_pos, int k);
        };
    }
}