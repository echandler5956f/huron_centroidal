#include <string>
#include <map>

namespace acrobatics
{
    namespace contact
    {
        struct EndEffector
        {
            // The name of the end effector frame in pinocchio. Used as the key pair in global end effector maps.
            std::string frame_name;
            // The id of the frame in pinocchio
            int frame_id;

            // Is 6-DOF or is 3-DOF
            bool is_6d;
        };

        // We use a ptr so that there is only ever one instance of the end effector.
        typedef RobotEndEffectors std::map<std::string, std::shared_ptr<EndEffector>>;

        typedef ContactCombination std::map<std::string, bool>;
    }
}