#pragma once

#include "Model/EndEffector.h"

#include <pinocchio/multibody/model.hpp>
#include <vector>
namespace acro
{
    namespace model
    {
        class LeggedBody : public pinocchio::Model
        {
        public:
            LeggedBody() : pinocchio::Model() {}

            // Provide the string IDs that correspond to the pinocchio end effector frames.
            void setEndEffectors(const std::vector<std::string> &ee_names);

            // Generate combinations of contacts.
            void GenerateContactCombination();

            // ContactCombination getContactCombination(const std::string&);
            
            // Referenced by binary value instead
            ContactCombination getContactCombination(int contact_mask){return contact_combinations[contact_mask];}

        private:
            std::vector<std::string> ee_names_;
            // Contact combination "1 2 3", where ee's 1,2, and 3 are in contact is at index with
            // a binary value of (1110) for a system with 4 EEs
            std::vector<ContactCombination> contact_combinations_;
            RobotEndEffectors ees_;
            int num_end_effectors_;
        };
    }
}