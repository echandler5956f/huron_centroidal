#include "Variables/PseudospectralSegment.h"

using namespace casadi;

namespace acro
{
    namespace variables
    {
        LagrangePolynomial::LagrangePolynomial(int d_, const std::string &scheme)
        {
            assert((d_ > 0) && (d_ < 10) && "Collocation degrees must be positive and no greater than 9");
            this->d = d_;
            /*Choose collocation points*/
            std::vector<double> troot = collocation_points(this->d, scheme);
            troot.insert(troot.begin(), 0);
            this->tau_root = Eigen::Map<Eigen::VectorXd>(troot.data(), troot.size());

            /*Coefficients of the quadrature function*/
            this->B.resize(this->d + 1);

            /*Coefficients of the collocation equation*/
            this->C.resize(this->d + 1, this->d + 1);

            /*Coefficients of the continuity equation*/
            this->D.resize(this->d + 1);

            /*For all collocation points*/
            for (int j = 0; j < this->d + 1; ++j)
            {
                /*Construct Lagrange polynomials to get the polynomial basis at the collocation point*/
                Polynomial p = 1;
                for (int r = 0; r < this->d + 1; ++r)
                {
                    if (r != j)
                    {
                        p *= Polynomial(-this->tau_root(r), 1) / (this->tau_root(j) - this->tau_root(r));
                    }
                }
                /*Evaluate the polynomial at the final time to get the coefficients of the continuity equation*/
                this->D(j) = p(1.0);

                /*Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation*/
                Polynomial dp = p.derivative();
                for (int r = 0; r < this->d + 1; ++r)
                {
                    this->C(j, r) = dp(this->tau_root(r));
                }
                Polynomial pint = p.anti_derivative();
                this->B(j) = pint(1.0);
            }
        }

        const SX LagrangePolynomial::lagrange_interpolation(double t, const SXVector terms)
        {
            assert((t >= 0.0) && (t <= 1.0) && "t must be in the range [0,1]");
            SX result = 0;
            for (int j = 0; j < this->d; ++j)
            {
                SX term = terms[j];
                for (int r = 0; r < this->d + 1; ++r)
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

        PseudospectralSegment::PseudospectralSegment(int d, int knot_num_, double h_, States *st_m_, Function &Fint_)
        {
            this->knot_num = knot_num_;
            this->Fint = Fint_;
            this->h = h_;
            this->st_m = st_m_;
            this->T = (this->knot_num + 1) * this->h;
            this->initialize_expression_variables(d);
            this->initialize_time_vector();
        }

        void PseudospectralSegment::initialize_expression_variables(int d)
        {
            this->dXc.clear();
            this->Uc.clear();

            this->dX_poly = LagrangePolynomial(d);
            this->U_poly = LagrangePolynomial(d - 1);

            for (int j = 0; j < d; ++j)
            {
                this->dXc.push_back(SX::sym("dXc_" + std::to_string(j), this->st_m->ndx, 1));
                if (j < d - 1)
                {
                    this->Uc.push_back(SX::sym("Uc_" + std::to_string(j), this->st_m->nu, 1));
                }
            }
            this->dX0 = SX::sym("dX0", this->st_m->ndx, 1);
            this->X0 = SX::sym("X0", this->st_m->nx, 1);
            this->Lc = SX::sym("Lc", 1, 1);
        }

        void PseudospectralSegment::initialize_time_vector()
        {
            this->times = DM::zeros(this->knot_num * (this->dX_poly.d + 1) + 1, 1);
            this->times(this->knot_num * (this->dX_poly.d + 1)) = this->T;
            for (int k = 0; k < this->knot_num; ++k)
            {
                for (int j = 0; j < this->dX_poly.d + 1; ++j)
                {
                    this->times(k * (this->dX_poly.d + 1) + j) = this->dX_poly.tau_root[j] * this->h + k * this->h;
                }
            }
        }

        void PseudospectralSegment::fill_times(std::vector<double> &all_times)
        {
            std::vector<double> element_access1 = this->times.get_elements();
            all_times.insert(all_times.end(), element_access1.begin(), element_access1.end());
        }

        void PseudospectralSegment::initialize_knot_segments(SX x0)
        {
            this->dXc_var_vec.clear();
            this->U_var_vec.clear();
            this->dX0_var_vec.clear();
            this->X0_var_vec.clear();
            for (int k = 0; k < this->knot_num; ++k)
            {
                this->dXc_var_vec.push_back(SX::sym("dXc_" + std::to_string(k), this->st_m->ndx * this->dX_poly.d, 1));
                this->U_var_vec.push_back(SX::sym("U_" + std::to_string(k) + "_", this->st_m->nu * this->U_poly.d, 1));
            }

            for (int k = 0; k < this->knot_num + 1; ++k)
            {
                this->dX0_var_vec.push_back(SX::sym("dX0_" + std::to_string(k), this->st_m->ndx, 1));
                this->X0_var_vec.push_back(this->Fint(SXVector{x0, this->dX0_var_vec[k], 1.0}).at(0));
            }
        }

        void PseudospectralSegment::initialize_expression_graph(Function &F, Function &L, std::vector<std::shared_ptr<ConstraintData>> G)
        {
            /*Collocation equations*/
            SXVector eq;
            /*State at the end of the collocation interval*/
            SX dXf = this->dX_poly.D(0) * this->dX0;
            /*Cost at the end of the collocation interval*/
            SX Qf = 0;
            /*Actual state at collocation points*/
            SXVector x_at_c;
            /*U interpolated at the dx polynomial collocation points*/
            SXVector u_at_c;

            for (int j = 1; j < this->dX_poly.d + 1; ++j)
            {
                double dt_j = this->dX_poly.tau_root(j) - this->dX_poly.tau_root(j - 1) * this->h;
                /*Expression for the state derivative at the collocation point*/
                SX dxp = this->dX_poly.C(0, j) * this->dX0;
                for (int r = 0; r < this->dX_poly.d; ++r)
                {
                    dxp += this->dX_poly.C(r + 1, j) * this->dXc[r];
                }
                /*dXc must exist in a Euclidean space, but we need x_c in order to evaluate the objective. Fint can simply return dXc[j-1] if the states are already Euclidean*/
                SX x_c = this->Fint(SXVector{this->X0, this->dXc[j - 1], dt_j}).at(0);
                SX u_c = this->U_poly.lagrange_interpolation(this->dX_poly.tau_root(j - 1), this->Uc);
                x_at_c.push_back(x_c);
                u_at_c.push_back(u_c);

                /*Append collocation equations*/
                eq.push_back(this->h * F(SXVector{x_c, u_c}).at(0) - dxp);

                /*Add cost contribution*/
                SXVector L_out = L(SXVector{x_c, u_c});
                /*This is fine as long as the cost is not related to the Lie Group elements. See the state integrator and dX for clarity*/
                Qf += this->dX_poly.B(j) * L_out.at(0) * this->h;
                // Qf += this->U_poly.B(j) * L_out.at(1) * this->h;

                dXf += this->dX_poly.D(j) * this->dXc[j - 1];
            }
            long N = this->knot_num * (2 + G.size());
            this->general_lb.resize(N, 1);
            this->general_ub.resize(N, 1);
            /*Implicit discrete-time equations*/
            this->collocation_constraint_map = Function("feq",
                                                        SXVector{this->X0, vertcat(this->dXc), this->dX0, vertcat(this->Uc)},
                                                        SXVector{vertcat(eq)})
                                                   .map(this->knot_num, "openmp");
            /*When you evaluate this map, subtract by the knot points list offset by 1 to be correct*/
            this->xf_constraint_map = Function("fxf",
                                               SXVector{this->X0, vertcat(this->dXc), this->dX0, vertcat(this->Uc)},
                                               SXVector{dXf})
                                          .map(this->knot_num, "openmp");

            this->general_lb(Slice(0, this->knot_num * 2)) = DM::zeros(this->knot_num * 2, 1);
            this->general_ub(Slice(0, this->knot_num * 2)) = DM::zeros(this->knot_num * 2, 1);

            this->q_cost_fold = Function("fxq",
                                         SXVector{this->Lc, this->X0, vertcat(this->dXc), this->dX0, vertcat(this->Uc)},
                                         SXVector{this->Lc + Qf})
                                    .fold(this->knot_num);
            /*Map the constraint to each collocation point, and then map the mapped constraint to each knot segment*/
            SXVector tmp_dx = this->dXc;
            tmp_dx.push_back(this->dX0); /*If we are doing this for state, is the size right for U?*/

            SXVector tmp_x = x_at_c;
            tmp_x.push_back(this->X0); /*If we are doing this for state, is the size right for U?*/

            for (int i = 0; i < G.size(); ++i)
            {
                auto g_data = G[i];
                SXVector tmp_map = g_data->F.map(this->dX_poly.d, "serial")(SXVector{vertcat(tmp_x), vertcat(u_at_c)});
                this->general_constraint_maps.push_back(Function("fg",
                                                                 SXVector{this->X0, vertcat(tmp_dx), vertcat(this->Uc)},
                                                                 SXVector{vertcat(tmp_map)})
                                                            .map(this->knot_num, "openmp"));

                this->general_lb(Slice(this->knot_num * (i + 2), this->knot_num * (i + 1 + 2))) =
                    vertcat(g_data->lower_bound.map(this->knot_num, "serial")(this->times));
                this->general_ub(Slice(this->knot_num * (i + 2), this->knot_num * (i + 1 + 2))) =
                    vertcat(g_data->upper_bound.map(this->knot_num, "serial")(this->times));
            }
        }

        void PseudospectralSegment::evaluate_expression_graph(SX &J0, SXVector &g)
        {
            SXVector result;
            auto xs_offset = this->X0_var_vec[1];
            auto dxs_offset = this->dX0_var_vec[1];
            auto xs = this->X0_var_vec[0];
            auto dxs = this->dX0_var_vec[0];
            auto us = this->U_var_vec[0];
            auto dxcs = this->dXc_var_vec[0];
            for (int k = 1; k < this->knot_num; ++k)
            {
                xs_offset = horzcat(xs_offset, this->X0_var_vec[k + 1]);
                dxs_offset = horzcat(dxs_offset, this->dX0_var_vec[k + 1]);
                xs = horzcat(xs, this->X0_var_vec[k]);
                dxs = horzcat(dxs, this->dX0_var_vec[k]);
                dxcs = horzcat(dxcs, this->dXc_var_vec[k]);
                us = horzcat(us, this->U_var_vec[k]);
            }
            result.push_back(this->collocation_constraint_map(SXVector{xs, dxcs, dxs, us}).at(0));

            result.push_back(this->xf_constraint_map(SXVector{xs, dxcs, dxs, us}).at(0) -
                             dxs_offset);

            for (std::size_t i = 0; i < this->general_constraint_maps.size(); ++i)
            {
                result.push_back(this->general_constraint_maps[i](SXVector{xs, dxcs, dxs, us}).at(0));
            }

            // Worried about aliasing here
            SX cost = this->q_cost_fold(SXVector{J0, xs, dxcs, dxs, us}).at(0);
            J0 = cost;
            /*where g of this segment starts*/
            auto tmp_it = g.end() + 1;
            g.insert(g.end(), result.begin(), result.end());
            this->g_range = tuple_size_t(tmp_it - g.begin(), g.end() - g.begin());
        }

        tuple_size_t PseudospectralSegment::get_range_idx_constraint_expressions()
        {
            return this->g_range;
        }

        SX PseudospectralSegment::get_initial_state_deviant()
        {
            return this->dX0_var_vec.front();
        }

        SX PseudospectralSegment::get_final_state_deviant()
        {
            return this->dX0_var_vec.back();
        }

        SX PseudospectralSegment::get_final_state()
        {
            return this->X0_var_vec.back();
        }

        void PseudospectralSegment::fill_lb_ub(std::vector<double> &lb, std::vector<double> &ub)
        {
            /*where lb/ub of this segment starts*/
            auto tmp_it = lb.end() + 1;
            std::vector<double> element_access1 = this->general_lb.get_elements();
            std::vector<double> element_access2 = this->general_ub.get_elements();

            lb.insert(lb.end(), element_access1.begin(), element_access1.end());
            ub.insert(ub.end(), element_access2.begin(), element_access2.end());
            this->lb_ub_range = tuple_size_t(tmp_it - lb.begin(), lb.end() - lb.begin());
        }

        tuple_size_t PseudospectralSegment::get_range_idx_bounds()
        {
            return this->lb_ub_range;
        }

        void PseudospectralSegment::fill_w(SXVector &w)
        {
            /*where w of this segment starts*/
            auto tmp_it = w.end() + 1;
            w.insert(w.end(), this->dXc_var_vec.begin(), this->dXc_var_vec.end());
            w.insert(w.end(), this->dX0_var_vec.begin(), this->dX0_var_vec.end());
            w.insert(w.end(), this->U_var_vec.begin(), this->U_var_vec.end());
            this->w_range = tuple_size_t(tmp_it - w.begin(), w.end() - w.begin());
        }

        tuple_size_t PseudospectralSegment::get_range_idx_decision_variables()
        {
            return this->w_range;
        }

    }
}