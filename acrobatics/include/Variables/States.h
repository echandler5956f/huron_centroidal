#pragma once

#include <pinocchio/autodiff/casadi.hpp>

namespace acro
{
    namespace variables
    {
        class States
        {
        public:
            States();
            States(const int nq_, const int nv_);

            static const int nu = 6;
            static const int nh = 6;
            static const int ndh = 12;
            static const int nqb = 7;
            static const int nvb = 6;

            /*Momenta: nh x 1*/
            template <class Sym>
            Sym get_ch(Sym cx)
            {
                return cx(casadi::Slice(0, this->nh));
            }
            /*Momenta delta: nh x 1*/
            template <class Sym>
            Sym get_ch_d(Sym cdx)
            {
                return cdx(casadi::Slice(0, this->ndh));
            }
            /*Momenta time derivative: nh x 1*/
            template <class Sym>
            Sym get_cdh(Sym cx)
            {
                return cx(casadi::Slice(this->nh, this->ndh));
            }
            /*Momentum time derivative delta: nh x 1*/
            template <class Sym>
            Sym get_cdh_d(Sym cdx)
            {
                return cdx(casadi::Slice(this->nh, this->ndh));
            }
            /*q: nq x 1*/
            template <class Sym>
            Sym get_q(Sym cx)
            {
                return cx(casadi::Slice(this->ndh, this->ndh + this->nq));
            }
            /*q delta: nv x 1*/
            template <class Sym>
            Sym get_q_d(Sym cdx)
            {
                return cdx(casadi::Slice(this->ndh, this->ndh + this->nv));
            }
            /*qj: (nq - 7) x 1*/
            template <class Sym>
            Sym get_qj(Sym cx)
            {
                return cx(casadi::Slice(this->ndh + this->nqb, this->ndh + this->nq));
            }
            /*v: nv x 1*/
            template <class Sym>
            Sym get_v(Sym cx)
            {
                return cx(casadi::Slice(this->ndh + this->nq, this->nx));
            }
            /*v delta: nv x 1*/
            template <class Sym>
            Sym get_v_d(Sym cdx)
            {
                return cdx(casadi::Slice(this->ndh + this->nv, this->ndx));
            }
            /*v_j: (nv - 6) x 1*/
            template <class Sym>
            Sym get_vj(Sym cx)
            {
                return cx(casadi::Slice(this->ndh + this->nq + this->nvb, this->nx));
            }

            int nq;
            int nv;
            int nx;
            int ndx;
        };
    }
}