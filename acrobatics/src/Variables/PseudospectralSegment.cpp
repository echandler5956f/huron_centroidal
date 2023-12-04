#include "Variables/PseudospectralSegment.h"

namespace acro
{
    namespace variables
    {
        LagrangePolynomial::LagrangePolynomial(int d_, const std::string &scheme)
        {
            assert((d_ > 0) && (d_ < 10) && "Collocation degrees must be positive and no greater than 9");
            this->d = d_;
            /*Choose collocation points*/
            auto troot = casadi::collocation_points(this->d, scheme);
            troot.insert(troot.begin(), 0);
            this->tau_root = Eigen::Map<Eigen::VectorXd>(troot.data(), troot.size());

            /*Coefficients of the quadrature function*/
            this->B.resize(this->d + 1);

            /*Coefficients of the collocation equation*/
            this->C.resize(this->d + 1, this->d + 1);

            /*Coefficients of the continuity equation*/
            this->D.resize(this->d + 1);

            /*For all collocation points*/
            for (auto j = 0; j < this->d + 1; ++j)
            {
                /*Construct Lagrange polynomials to get the polynomial basis at the collocation point*/
                casadi::Polynomial p = 1;
                for (auto r = 0; r < this->d + 1; ++r)
                {
                    if (r != j)
                    {
                        p *= casadi::Polynomial(-this->tau_root(r), 1) / (this->tau_root(j) - this->tau_root(r));
                    }
                }
                /*Evaluate the polynomial at the final time to get the coefficients of the continuity equation*/
                this->D(j) = p(1.0);

                /*Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation*/
                casadi::Polynomial dp = p.derivative();
                for (auto r = 0; r < this->d + 1; ++r)
                {
                    this->C(j, r) = dp(this->tau_root(r));
                }
                auto pint = p.anti_derivative();
                this->B(j) = pint(1.0);
            }
        }

        const casadi::SX LagrangePolynomial::lagrange_interpolation(double t, const std::vector<casadi::SX> terms)
        {
            assert((t >= 0.0) && (t <= 1.0) && "t must be in the range [0,1]");
            casadi::SX result = 0;
            for (auto j = 0; j < this->d + 1; ++j)
            {
                casadi::SX term = terms[j];
                for (auto r = 0; r < this->d + 1; ++r)
                {
                    if (r != j)
                    {
                        term *= (t - this->tau_root(r)) / (this->tau_root(j) - this->tau_root(r));
                    }
                }
                result += term;
            }
            return result;
        }

        PseudospectralSegment::PseudospectralSegment(int d, int knot_num_, double h_, States *st_m_, casadi::Function &Fint_)
        {
            this->knot_num = knot_num_;
            this->Fint = Fint_;
            this->h = h_;
            this->st_m = st_m_;
            this->T = (this->knot_num + 1) * this->h;
            this->initialize_expression_variables(d);
            this->initialize_time_vector();
            this->initialize_knot_segments();
        }

        void PseudospectralSegment::initialize_expression_variables(int d)
        {
            this->dXc.clear();
            this->Uc.clear();

            this->dX_poly = LagrangePolynomial(d);
            this->U_poly = LagrangePolynomial(d - 1);

            for (auto j = 0; j < d; ++j)
            {
                this->dXc.push_back(casadi::SX::sym("dXc_" + std::to_string(j), this->st_m->ndx, 1));
                if (j < d - 1)
                {
                    this->Uc.push_back(casadi::SX::sym("Uc_" + std::to_string(j), this->st_m->nu, 1));
                }
            }
            this->dX0 = casadi::SX::sym("dX0", this->st_m->ndx, 1);
            this->Lc = casadi::SX::sym("Lc", 1, 1);
        }

        void PseudospectralSegment::initialize_time_vector()
        {
            this->times = casadi::DM::zeros(this->knot_num * (this->dX_poly.d + 1) + 1, 1);
            this->times(this->knot_num * (this->dX_poly.d + 1)) = this->T;
            for (auto k = 0; k < this->knot_num; ++k)
            {
                for (auto j = 0; j < this->dX_poly.d + 1; ++j)
                {
                    this->times(k * (this->dX_poly.d + 1) + j) = this->dX_poly.tau_root[j] * this->h + k * this->h;
                }
            }
        }

        void PseudospectralSegment::initialize_knot_segments()
        {
            this->traj_segment.clear();
            for (auto k = 0; k < this->knot_num; ++k)
            {
                KnotSegment knot_segment_k;
                for (auto j = 0; j < this->dX_poly.d; ++j)
                {
                    knot_segment_k.dXc_var.push_back(casadi::SX::sym("dXc_" + std::to_string(k) + "_" + std::to_string(j), this->st_m->ndx, 1));
                }
                for (auto j = 0; j < this->U_poly.d; ++j)
                {
                    knot_segment_k.U_var.push_back(casadi::SX::sym("U_" + std::to_string(k) + "_" + std::to_string(j), this->st_m->nu, 1));
                }
                knot_segment_k.dX0_var = casadi::SX::sym("dX0_" + std::to_string(k), this->st_m->ndx, 1);
                this->traj_segment.push_back(knot_segment_k);
            }
        }

        void PseudospectralSegment::initialize_expression_graph(casadi::Function &F, casadi::Function &L, std::vector<std::shared_ptr<ConstraintData>> G)
        {
            /*Collocation equations*/
            std::vector<casadi::SX> eq;
            /*State at the end of the collocation interval*/
            casadi::SX dXf = this->dX_poly.D(0) * this->dX0;
            /*Cost at the end of the collocation interval*/
            casadi::SX Qf = 0;
            /*U interpolated at the dx polynomial collocation points*/
            std::vector<casadi::SX> u_at_c;

            for (auto j = 1; j < this->dX_poly.d + 1; ++j)
            {
                double dt_j = this->dX_poly.tau_root(j) - this->dX_poly.tau_root(j - 1) * this->h;
                /*Expression for the state derivative at the collocation point*/
                casadi::SX dxp = this->dX_poly.C(0, j) * this->dX0;
                for (auto r = 0; r < this->dX_poly.d; ++r)
                {
                    dxp += this->dX_poly.C(r + 1, j) * this->dXc[r];
                }

                casadi::SX x_c = this->Fint(std::vector<casadi::SX>{this->dX0, this->dXc[j - 1], dt_j}).at(0);
                casadi::SX u_c = this->U_poly.lagrange_interpolation(this->dX_poly.tau_root(j - 1), this->Uc);
                u_at_c.push_back(u_c);

                /*Append collocation equations*/
                eq.push_back(this->h * F(std::vector<casadi::SX>{x_c, u_c}).at(0) - dxp);

                /*Add cost contribution*/
                std::vector<casadi::SX> L_out = L(std::vector<casadi::SX>{x_c, u_c});
                /*This is fine as long as the cost is not related to the Lie Group elements. See the state integrator and dX for clarity*/
                Qf += this->dX_poly.B(j) * L_out.at(0) * this->h;
                Qf += this->U_poly.B(j) * L_out.at(1) * this->h;

                dXf += this->dX_poly.D(j) * this->dXc[j - 1];
            }
            long N = this->knot_num * (2 + G.size());
            this->general_lb.resize(N, 1);
            this->general_ub.resize(N, 1);
            /*Implicit discrete-time equations*/
            this->collocation_constraint_map = casadi::Function("feq",
                                                                std::vector<casadi::SX>{vertcat(this->dXc), this->dX0, vertcat(this->Uc)},
                                                                std::vector<casadi::SX>{vertcat(eq)})
                                                   .map(this->knot_num, "openmp");
            this->xf_constraint_map = casadi::Function("fxf",
                                                       std::vector<casadi::SX>{vertcat(this->dXc), this->dX0, vertcat(this->Uc)},
                                                       std::vector<casadi::SX>{dXf})
                                          .map(this->knot_num, "openmp");

            this->general_lb(casadi::Slice(0, this->knot_num * 2)) = casadi::DM::zeros(this->knot_num * 2, 1);
            this->general_ub(casadi::Slice(0, this->knot_num * 2)) = casadi::DM::zeros(this->knot_num * 2, 1);

            this->q_cost_fold = casadi::Function("fxq",
                                                 std::vector<casadi::SX>{this->Lc, vertcat(this->dXc), this->dX0, vertcat(this->Uc)},
                                                 std::vector<casadi::SX>{this->Lc + Qf})
                                    .fold(this->knot_num);
            /*Map the constraint to each collocation point, and then map the mapped constraint to each knot segment*/
            casadi::SXVector tmp_dx = this->dXc;
            tmp_dx.push_back(this->dX0); /*If we are doing this for state, is the size right for U?*/

            for (auto i = 0; i < G.size(); ++i)
            {
                auto g_data = G[i];
                casadi::SXVector tmp_map = g_data->F.map(this->dX_poly.d, "serial")(std::vector<casadi::SX>{vertcat(tmp_dx), vertcat(u_at_c)});
                this->general_constraint_maps.push_back(casadi::Function("fg",
                                                                         std::vector<casadi::SX>{vertcat(tmp_dx), vertcat(this->Uc)},
                                                                         std::vector<casadi::SX>{vertcat(tmp_map)})
                                                            .map(this->knot_num, "openmp"));

                this->general_lb(casadi::Slice(this->knot_num * (i + 2), this->knot_num * (i + 1 + 2))) = vertcat(g_data->lower_bound.map(this->knot_num, "serial")(this->times));
                this->general_ub(casadi::Slice(this->knot_num * (i + 2), this->knot_num * (i + 1 + 2))) = vertcat(g_data->upper_bound.map(this->knot_num, "serial")(this->times));
            }
        }
    }
}