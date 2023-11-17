#include "Model/EndEffector.h"
namespace acrobatics
{
    namespace contact
    {
        class ContactSequence
        {
        public:
            ContactSequence(int num_end_effectors) : num_end_effectors_(num_end_effectors) {}
            // we will fill this out as needed.
        private:
            int num_end_effectors;
        };
    }
}