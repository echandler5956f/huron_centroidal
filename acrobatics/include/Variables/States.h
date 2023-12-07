#pragma once

#include <pinocchio/autodiff/casadi.hpp>

namespace acro
{
    namespace variables
    {
        /**
         * @brief Simple slicer class for getting state variables
         *
         */
        class States
        {
        public:
            /**
             * @brief Construct a new States object
             *
             */
            States() {}

            /**
             * @brief Construct a new States object
             *
             * @param nq_ Number of position variables
             * @param nv_ Number of velocity variables
             */
            States(const int nq_, const int nv_);

            /**
             * @brief Input space dimension
             *
             */
            static const int nF = 6;

            /**
             * @brief Momenta space dimension
             *
             */
            static const int nh = 6;

            /**
             * @brief Momenta time derivative offset
             *
             */
            static const int ndh = 6;

            /**
             * @brief Number of position coordinates for the base
             *
             */
            static const int nqb = 7;

            /**
             * @brief Number of velocity coordinates for the base
             *
             */
            static const int nvb = 6;

            /**
             * @brief Get mMomenta: nh x 1
             *
             * @tparam Sym
             * @param cx
             * @return const Sym
             */
            template <class Sym>
            const Sym get_ch(const Sym &cx)
            {
                return cx(casadi::Slice(0, this->nh));
            }

            /**
             * @brief Get momenta delta: nh x 1
             *
             * @tparam Sym
             * @param cdx
             * @return const Sym
             */
            template <class Sym>
            const Sym get_ch_d(const Sym &cdx)
            {
                return cdx(casadi::Slice(0, this->nh));
            }

            /**
             * @brief Get momenta time derivative: nh x 1
             *
             * @tparam Sym
             * @param cx
             * @return const Sym
             */
            template <class Sym>
            const Sym get_cdh(const Sym &cx)
            {
                return cx(casadi::Slice(this->nh, this->nh + this->ndh));
            }

            /**
             * @brief Get momentum time derivative delta: nh x 1
             *
             * @tparam Sym
             * @param cdx
             * @return const Sym
             */
            template <class Sym>
            const Sym get_cdh_d(const Sym &cdx)
            {
                return cdx(casadi::Slice(this->nh, this->nh + this->ndh));
            }

            /**/
            /**
             * @brief Get q: nq x 1
             *
             * @tparam Sym
             * @param cx
             * @return const Sym
             */
            template <class Sym>
            const Sym get_q(const Sym &cx)
            {
                return cx(casadi::Slice(this->nh + this->ndh, this->nh + this->ndh + this->nq));
            }

            /**
             * @brief Get q delta: nv x 1
             *
             * @tparam Sym
             * @param cdx
             * @return const Sym
             */
            template <class Sym>
            const Sym get_q_d(const Sym &cdx)
            {
                return cdx(casadi::Slice(this->nh + this->ndh, this->nh + this->ndh + this->nv));
            }

            /**
             * @brief Get qj: (nq - 7) x 1
             *
             * @tparam Sym
             * @param cx
             * @return const Sym
             */
            template <class Sym>
            const Sym get_qj(const Sym &cx)
            {
                return cx(casadi::Slice(this->nh + this->ndh + this->nqb, this->nh + this->ndh + this->nq));
            }

            /**
             * @brief Get v: nv x 1
             *
             * @tparam Sym
             * @param cx
             * @return const Sym
             */
            template <class Sym>
            const Sym get_v(const Sym &cx)
            {
                return cx(casadi::Slice(this->nh + this->ndh + this->nq, this->nx));
            }

            /**
             * @brief Get v delta: nv x 1
             *
             * @tparam Sym
             * @param cdx
             * @return const Sym
             */
            template <class Sym>
            const Sym get_v_d(const Sym &cdx)
            {
                return cdx(casadi::Slice(this->nh + this->ndh + this->nv, this->ndx));
            }

            /**
             * @brief Get v_j: (nv - 6) x 1
             *
             * @tparam Sym
             * @param cx
             * @return const Sym
             */
            template <class Sym>
            const Sym get_vj(const Sym &cx)
            {
                return cx(casadi::Slice(this->nh + this->ndh + this->nq + this->nvb, this->nx));
            }

            /**
             * @brief Get f: 3 x 1
             *
             * @tparam Sym
             * @param u
             * @return const Sym
             */
            template <class Sym>
            const Sym get_f(const Sym &u)
            {
                return u(casadi::Slice(0, 3));
            }

            /**
             * @brief Get tau: 3 x 1
             *
             * @tparam Sym
             * @param u
             * @return const Sym
             */
            template <class Sym>
            const Sym get_tau(const Sym &u)
            {
                return u(casadi::Slice(3, this->nF));
            }

            /**
             * @brief Get tau: nvju x 1
             *
             * @tparam Sym
             * @param u
             * @return const Sym
             */
            template <class Sym>
            const Sym get_vju(const Sym &u)
            {
                return u(casadi::Slice(this->nF, this->nu));
            }

            /**
             * @brief Number of position variables
             *
             */
            int nq;

            /**
             * @brief Number of velocity variables
             *
             */
            int nv;

            /**
             * @brief Number of state variables
             *
             */
            int nx;

            /**
             * @brief Number of state derivative variables
             *
             */
            int ndx;

            /**
             * @brief Number of input variables (nF + nvju)
             *
             */
            int nu;

            /**
             * @brief Number of joint velocity inputs
             * 
             */
            int nvju;
        };
    }
}