#include <Eigen/Core>
#include <pinocchio/autodiff/casadi.hpp>

namespace constraint
{

    // The actual implementation is on another branch
    struct ProblemData
    {
    };

    struct ConstraintData
    {
        Eigen::VectorXi flags;

        // Shoud this be one vector of upper & lower bound, or a casadi function of time to map
        Eigen::VectorXd upper_bound;
        Eigen::VectorXd lower_bound;

        casadi::Function F;
    };

    class CustomConstraint
    {
        CustomConstraint() {}

        virtual ~CustomConstraint() = default;

        void BuildConstraint(const ProblemData &problem_data, ConstraintData &constraint_data)
        {
            CreateFlags(problem_data, constraint_data.flags);
            CreateBounds(problem_data, constraint_data.upper_bound, constraint_data.lower_bound);
            CreateFunction(problem_data, constraint_data.F);
        }

        // Generate flags for each collocation point
        virtual void CreateFlags(const ProblemData &problem_data, Eigen::VectorXi &flags) const;

        // Generate bounds for a vector of concatinated collocation points
        virtual void CreateBounds(const ProblemData &problem_data, Eigen::VectorXd &upper_bound, Eigen::VectorXd &lower_bound) const;

        // Generate a function to evaluate each collocation point.
        virtual void CreateFunction(const ProblemData &problem_data, casadi::Function &F) const;
    };

};