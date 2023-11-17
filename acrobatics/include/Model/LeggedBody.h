#include "Model/EndEffector.h"

#include <pinocchio/multibody/model.hpp>
#include <vector>
namespace acrobatics
{
    namespace contact
    {
        class LeggedBody : public pinocchio::Model
        {
        public:
            LeggedBody() : pinocchio::Model() {}

            // Provide the string IDs that correspond to the pinocchio end effector frames.
            void setEndEffectors(const std::vector<std::string> &ee_names);

            // Generate combinations of contacts.
            void GenerateContactCombination();

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