#include "Variables/PseudospectralSegment.h"

void acro::variables::LagrangePolynomial::compute_matrices(int d_, const std::string &scheme)
{
    this->d = d_;
    /*Choose collocation points*/
    auto troot = casadi::collocation_points(d, scheme);
    troot.insert(troot.begin(), 0);
    this->tau_root = Eigen::Map<Eigen::VectorXd>(troot.data(), troot.size());

    /*Coefficients of the quadrature function*/
    this->B.resize(d + 1);

    /*Coefficients of the collocation equation*/
    this->C.resize(d + 1, d + 1);

    /*Coefficients of the continuity equation*/
    this->D.resize(d + 1);

    /*For all collocation points*/
    for (int j = 0; j < d + 1; ++j)
    {
        /*Construct Lagrange polynomials to get the polynomial basis at the collocation point*/
        casadi::Polynomial p = 1;
        for (int r = 0; r < this->d + 1; ++r)
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
        for (int r = 0; r < d + 1; ++r)
        {
            this->C(j, r) = dp(this->tau_root(r));
        }
        auto pint = p.anti_derivative();
        this->B(j) = pint(1.0);
    }
}

casadi::SX acro::variables::LagrangePolynomial::lagrange_interpolation(double t, std::vector<casadi::SX> terms)
{
    casadi::SX result = 0;
    for (int j = 0; j < d + 1; ++j)
    {
        casadi::SX term = terms[j];
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

acro::variables::PseudospectralSegment::PseudospectralSegment(int d, int knot_num_, double h_, States *st_m_)
{
    this->knot_num = knot_num_;
    this->h = h_;
    this->st_m = st_m_;
    this->initialize_expression_variables(d);
}

void acro::variables::PseudospectralSegment::initialize_expression_variables(int d)
{
    this->Xc.clear();
    this->Uc.clear();

    this->X_poly.compute_matrices(d);
    this->U_poly.compute_matrices(d - 1);

    for (int j = 0; j < d; ++j)
    {
        this->Xc.push_back(casadi::SX::sym("Xc_" + std::to_string(j), this->st_m->ndx, 1));
        if (j < d - 1)
        {
            this->Uc.push_back(casadi::SX::sym("Uc_" + std::to_string(j), this->st_m->nu, 1));
        }
    }
    this->X0 = casadi::SX::sym("X0", this->st_m->ndx, 1);
    this->Lc = casadi::SX::sym("Lc", 1, 1);
}

void acro::variables::PseudospectralSegment::initialize_expression_graph(casadi::Function F_, casadi::Function L_x, casadi::Function L_u, casadi::Function Fint)
{

    // Collocation equations
    std::vector<casadi::SX> eq;
    // State at the end of the collocation interval
    casadi::SX Xf = this->X_poly.D(0) * this->X0;
    // Cost at the end of the collocation interval
    casadi::SX Qf = 0;

    for (int j = 1; j < this->X_poly.d + 1; ++j)
    {
        double dt_j = this->X_poly.tau_root(j) - this->X_poly.tau_root(j - 1) * h;
        // Expression for the state derivative at the collocation point
        casadi::SX xp = this->X_poly.C(0, j) * this->X0;
        for (int r = 0; r < this->X_poly.d; ++r)
        {
            xp += this->X_poly.C(r + 1, j) * this->Xc[r];
        }

        casadi::SX x_c = Fint(std::vector<casadi::SX>{this->X0, this->Xc[j - 1], dt_j}).at(0);
        casadi::SX u_c = this->U_poly.lagrange_interpolation(this->X_poly.tau_root(j - 1), this->Uc);

        // Append collocation equations
        eq.push_back(this->h * F_(std::vector<casadi::SX>{x_c, u_c}).at(0) - xp);

        // Add cost contribution
        Qf += this->X_poly.B(j) * L_x(std::vector<casadi::SX>{Xc[j]}).at(0) * h;
        Qf += this->U_poly.B(j) * L_u(std::vector<casadi::SX>{u_c}).at(0) * h;

        Xf += this->X_poly.D(j) * this->Xc[j - 1];
    }

    // // Implicit discrete-time dynamics
    // casadi::Function Feq("feq", {vertcat(Xc), X0, P}, {vertcat(eq)});
    // casadi::Function Fxf("feq", {vertcat(Xc), X0, P}, {Xf});
    // casadi::Function Fxq("fxq", {Lc, vertcat(Xc), X0, P}, {Lc + Qf});
}