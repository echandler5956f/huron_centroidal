#pragma once

#include "Variables/States.h"

namespace acro
{
    namespace variables
    {
        /*Much of this class was borrowed from OCS2*/
        enum class ConstraintOrder
        {
            Linear,
            Quadratic
        };

        class StateInputConstraint
        {
        public:
            explicit StateInputConstraint(ConstraintOrder order) : order_(order) {}
            virtual ~StateInputConstraint() = default;

            /** Get the constraint order (Linear or Quadratic) */
            constexpr ConstraintOrder getOrder() const { return order_; };

            /** Get the constraint linear approximation */
            virtual casadi::Function getLinearApproximation(const casadi::Function &F, const casadi::SX &x, const casadi::SX &u, const casadi::SX &x0, const casadi::SX &u0) const
            {
                if (order_ == ConstraintOrder::Linear)
                {
                    casadi::SX x_lim = casadi::SX::sym("x_lim", x.sparsity());
                    casadi::SX u_lim = casadi::SX::sym("u_lim", u.sparsity());
                    casadi::SX f_linx = casadi::SX::substitute(F.sx_out().at(0) + casadi::SX::jtimes(F.sx_out().at(0), x, x_lim - x0), casadi::SX::vertcat(std::vector<casadi::SX>{x_lim, x}), casadi::SX::vertcat(std::vector<casadi::SX>{x, x0}));
                    casadi::SX u_linx = casadi::SX::substitute(f_linx + casadi::SX::jtimes(f_linx, u, u_lim - u0), casadi::SX::vertcat(std::vector<casadi::SX>{u_lim, u}), casadi::SX::vertcat(std::vector<casadi::SX>{u, u0}));
                    return casadi::Function("Linearized Function", std::vector<casadi::SX>{x, u, x0, u0}, std::vector<casadi::SX>{u_linx});
                }
                else
                {
                    throw std::runtime_error("[StateInputConstraint] The class only provides Quadratic approximation! call getQuadraticApproximation()");
                }
            }

            /** Get the constraint quadratic approximation */
            virtual casadi::Function getQuadraticApproximation(const casadi::SX &rhs, const casadi::SX &state, const casadi::SX &input, const casadi::SX &state0, const casadi::SX &input0) const
            {
                if (order_ == ConstraintOrder::Quadratic)
                {
                    throw std::runtime_error("[StateConstraint] Quadratic approximation not implemented!");
                }
                else
                {
                    throw std::runtime_error("[StateConstraint] The class only provides Linear approximation! call getLinearApproximation()");
                }
            }

        protected:
            StateInputConstraint(const StateInputConstraint &rhs) = default;

        private:
            ConstraintOrder order_;
        };

        class LinearStateInputConstraint : public StateInputConstraint
        {
        public:
            ~LinearStateInputConstraint() override = default;

            casadi::Function getLinearApproximation(const casadi::Function &F, const casadi::SX &x, const casadi::SX &u, const casadi::SX &x0, const casadi::SX &u0) const final;
        };
    }
}