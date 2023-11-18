#include "Variables/PseudospectralSegment.h"

acro::variables::PseudospectralSegment::PseudospectralSegment()
{
}

void acro::variables::PseudospectralSegment::compute_collocation_matrices(int d, Eigen::VectorXd &B, Eigen::MatrixXd &C, Eigen::VectorXd &D, const std::string &scheme)
{
    /*Choose collocation points*/
    auto troot = casadi::collocation_points(d, scheme);
    troot.insert(troot.begin(), 0);
    Eigen::VectorXd tau_root = Eigen::Map<Eigen::VectorXd>(troot.data(), troot.size());

    /*Coefficients of the quadrature function*/
    B.resize(d + 1);

    /*Coefficients of the collocation equation*/
    C.resize(d + 1, d + 1);

    /*Coefficients of the continuity equation*/
    D.resize(d + 1);

    /*For all collocation points*/
    for (int j = 0; j < d + 1; ++j)
    {

        /*Construct Lagrange polynomials to get the polynomial basis at the collocation point*/
        casadi::Polynomial p = 1;
        for (int r = 0; r < d + 1; ++r)
        {
            if (r != j)
            {
                p *= casadi::Polynomial(-tau_root(r), 1) / (tau_root(j) - tau_root(r));
            }
        }
        /*Evaluate the polynomial at the final time to get the coefficients of the continuity equation*/
        D(j) = p(1.0);

        /*Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation*/
        casadi::Polynomial dp = p.derivative();
        for (int r = 0; r < d + 1; ++r)
        {
            C(j, r) = dp(tau_root(r));
        }
        auto pint = p.anti_derivative();
        B(j) = pint(1.0);
    }
}

// acro::variables::PseudospectralSegment::PseudospectralSegment(contact::ContactCombination contact_combination)
//     : contact_combination_(contact_combination)
// {
//     InitMask();
//     InitConstraints();
// }

// void acro::variables::PseudospectralSegment::InitMask()
// {
//     int num_end_effectors = contact_combination_.size();
//     contact_mask_ = Eigen::VectorXd::Zero(num_end_effectors);
//     // TODO contact_combination_ expects a string key, not an int
//     for (int i = 0; i < num_end_effectors; i++)
//     {
//         contact_mask_[i] = contact_combination_[i].second;
//     }
// }

// void acro::variables::PseudospectralSegment::InitConstraints()
// {
// }