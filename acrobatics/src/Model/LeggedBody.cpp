#include "Model/EndEffector.h"

// class LeggedBody : public pinocchio::Model
// {
// public:
//     LeggedBody() : pinocchio::Model() {}

//     // Provide the string IDs that correspond to the pinocchio end effector frames.
//     void setEndEffectors(const std::vector<std::string> &ee_names);

//     // Generate combinations of contacts. "012" is a key that refers to a combination where end effectors 0, 1, and 2 are in contact.
//     std::map<std::string, ContactCombination> GenerateContactCombination();

// private:
//     std::vector<std::string> ee_names_;
//     RobotEndEffectors ees_;
//     int num_end_effectors_;
// };