#include "Model/LeggedBody.h"
#include "Variables/Trajectory.h"
#include <string>
using namespace acro;

const std::string huron_location = "/home/akshay/Documents/BiQu_acrobatics/src/huron_centroidal/resources/urdf/huron_cheat.urdf";
const int num_ees = 2;
const std::string end_effector_names[] = {"l_foot_v_ft_link", "r_foot_v_ft_link"};

void defineRobot(model::LeggedBody &bot)
{
    std::vector<std::string> ee_name_vect;
    ee_name_vect.resize(num_ees);
    for (int i = 0; i < num_ees; i++)
    {
        ee_name_vect[i] = end_effector_names[i]
    }

    pinocchio::urdf::buildModel(huron_location, bot);

    bot.setEndEffectors(ee_name_vect);
    bot.GenerateContactCombination();
}

int main()
{
    model::LeggedBody bot;
    defineRobot(bot);

    variables::Trajectory traj;
    //  Initializes dynamics
    traj.setModel(bot);

    // Target defines parameter for defining the cost
    // problemSetup defines parameters for setting up the problem, such as initial state
    variables::Target target;
    variables::ProblemSetup problem_setup;

    // A contact sequence has timing and knot metadata
    conact::ContactSequence contact_sequence;

    contact::generateContactSequence(contact_sequence, target, problem_setup);

    // traj.setContactSequence(contact_sequence);

    // traj.initProblem(problem_setup, target);

    // // defines what the solution looks like
    // variables::Solution solution;

    // traj.solve(solution);
}