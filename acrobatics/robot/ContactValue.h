#include <vector>
#include <Eigen/Core>

namespace acrobatics
{

    // A contact value states whichlegs are in contact with the ground, and data about it. We can figure out the specifics later.
    class ContactValue
    {
        ContactValue();

        std::map<int, bool> contact_map;
    };
}