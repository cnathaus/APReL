import aprel
import numpy as np
import gym
from aprel.querying.inquire import Inquire
from aprel.learning.inquire_learning import InquireLearning
import time
import itertools
import matplotlib.pyplot as plt


def feature_func(traj):
    """Returns the features of the given MountainCar trajectory, i.e. \Phi(traj).

    Args:
        traj: List of state-action tuples, e.g. [(state0, action0), (state1, action1), ...]

    Returns:
        features: a numpy vector corresponding the features of the trajectory
    """
    states = np.array([pair[0] for pair in traj])
    min_pos, max_pos = states[:, 0].min(), states[:, 0].max()
    mean_speed = np.abs(states[:, 1]).mean()
    mean_vec = [-0.703, -0.344, 0.007]
    std_vec = [0.075, 0.074, 0.003]
    return (np.array([min_pos, max_pos, mean_speed]) - mean_vec) / std_vec


def find_next_queries_inquire(belief, trajectory_set):
    curr_w = np.array([sample["weights"] for sample in belief.samples])
    trajectory_indices, gains = Inquire.generate_query(trajectory_set, curr_w)
    print("trajectory_indices: ", trajectory_indices)
    query = aprel.PreferenceQuery(trajectory_set[trajectory_indices])
    queries = [query]

    return queries, gains


def update_belief_inquire(queries, responses, belief, true_user):
    curr_w = np.array([sample["weights"] for sample in belief.samples])
    M = len(curr_w)
    w_dim = len(curr_w[0])
    rand = np.random.RandomState(0)

    # match the responses to the queries
    queries_with_responses = []
    for i, query in enumerate(queries):
        queries_with_responses.append(aprel.Preference(query, responses[i]))
    # if queries_with_responses doesn't contain all the queries, raise an error
    if len(queries_with_responses) != len(queries):
        raise ValueError("Not all queries have been responded to.")

    t_s = time.time()
    w_dist, w_opt = InquireLearning.gradient_descent(
        rand,
        queries_with_responses,
        1,
        # 0.8,
        w_dim,
        M,
        # prev_w=curr_w,  # TODO: see if this improves performance
    )
    t_e = time.time()
    print("Time to update belief: ", t_e - t_s)

    # Report the metrics
    belief.samples = [{"weights": w} for w in w_opt]

    return belief

    # if self._logging:
    #     iteration = len(self.queries)
    #     self.log_response(iteration, queries_with_responses[-1])
    #     self.log_belief(iteration)


def main():
    # Create the OpenAI Gym environment
    gym_env = gym.make("MountainCarContinuous-v0")

    use_inquire = True

    # Seed for reproducibility
    seed = 0
    np.random.seed(seed)
    gym_env.seed(seed)

    # Wrap the environment with a feature function
    env = aprel.GymEnvironment(gym_env, feature_func)

    num_trajectories = 100
    max_episode_length = 300
    num_iterations = 20
    num_samples = 100
    burnin = 200
    thin = 20
    proposal_distribution = aprel.gaussian_proposal

    # Create a trajectory set
    trajectory_set = aprel.generate_trajectories_randomly_gym(
        env,
        num_trajectories=num_trajectories,
        max_episode_length=max_episode_length,
        file_name="MountainCarContinuous-v0",
        restore=False,
        headless=True,
        seed=seed,
    )

    # plot the trajectory set feature differences

    trajectories = trajectory_set.trajectories

    trajectory_features = [trajectory.features for trajectory in trajectories]

    # normalize the features
    mean = np.mean(trajectory_features, axis=0)
    std = np.std(trajectory_features, axis=0)
    for trajectory in trajectories:
        trajectory.features = (trajectory.features - mean) / std

    # remake the trajectory set
    trajectory_set = aprel.TrajectorySet(trajectories)

    # Compute the query set
    subsets = np.array(
        [list(tup) for tup in itertools.combinations(np.arange(len(trajectories)), 2)]
    )

    # Compute the feature differences for each query
    feature_differences = []
    colors = []

    # discard the bad trajectories
    for subset in subsets:
        feat_diff = trajectories[subset[0]].features - trajectories[subset[1]].features
        feature_differences.append(feat_diff)
        colors.append("b")
    feature_differences = np.array(feature_differences)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        feature_differences[:, 0],
        feature_differences[:, 1],
        feature_differences[:, 2],
        c=colors,
        marker="o",
        s=1,
        label="3 features",
    )

    # labels
    ax.set_xlabel("Feature Difference 1")
    ax.set_ylabel("Feature Difference 2")
    ax.set_zlabel("Feature Difference 3")

    # make plot square with equal axes, centered at (0,0)
    ax.axis("square")

    # save plot
    plt.savefig("feature_differences_test_mountain_car.png", dpi=300)

    features_dim = len(trajectory_set[0].features)

    # Initialize the query optimizer
    query_optimizer = aprel.QueryOptimizerDiscreteTrajectorySet(trajectory_set)

    repeats = 5

    cos_sim_list_list = []

    for r in range(repeats):
        # Initialize the object for the true human
        true_params = {
            "weights": aprel.util_funs.get_random_normalized_vector(features_dim)
        }
        true_user = aprel.SoftmaxUser(true_params)

        # Create the human response model and initialize the belief distribution
        params = {"weights": aprel.util_funs.get_random_normalized_vector(features_dim)}
        user_model = aprel.SoftmaxUser(params)
        belief = aprel.SamplingBasedBelief(
            user_model,
            [],
            params,
            logprior=aprel.uniform_logprior,
            num_samples=num_samples,
            proposal_distribution=proposal_distribution,
            burnin=burnin,
            thin=thin,
        )
        # Report the metrics
        print("Estimated user parameters: " + str(belief.mean))
        cos_sim = aprel.cosine_similarity(belief, true_user)
        print("Cosine Similarity: " + str(cos_sim))

        queries = []
        responses = []

        cos_sim_list = []

        # Active learning loop
        for query_no in range(num_iterations):
            # Optimize the query
            if use_inquire:
                query, objective_values = find_next_queries_inquire(
                    belief, trajectory_set
                )
            else:
                query, objective_values = query_optimizer.optimize(
                    "mutual_information",
                    belief,
                    aprel.PreferenceQuery(trajectory_set[:2]),
                    batch_size=1,
                    optimization_method="exhaustive_search",
                    reduced_size=100,
                    gamma=1,
                    distance=aprel.default_query_distance,
                )

            # Ask the query to the human
            response = true_user.respond(query)

            query = query[0]
            response = response[0]

            queries.append(query)
            responses.append(response)
            print("Number of queries: " + str(len(queries)))

            if use_inquire:
                belief = update_belief_inquire(queries, responses, belief, true_user)
            else:
                # Update the belief distribution
                belief.update([aprel.Preference(query, response)])

            # Report the metrics
            print("Estimated user parameters: " + str(belief.mean))
            cos_sim = aprel.cosine_similarity(belief, true_user)
            print("Cosine Similarity: " + str(cos_sim))
            cos_sim_list.append(cos_sim)
        print("finished repeat: ", r + 1)
        cos_sim_list_list.append(cos_sim_list)

    cos_sim_list_list = np.array(cos_sim_list_list)

    # compute mean, std, round to 4 decimal places
    mean = np.mean(cos_sim_list_list, axis=0)
    std = np.std(cos_sim_list_list, axis=0)
    iterations = np.arange(num_iterations)
    data = np.vstack((iterations, mean, std)).T

    # save the mean and std to a txt file
    np.savetxt(
        "cos_sim_mountain_car.txt",
        data,
        header="iterations\tmean\tstd",
        delimiter="\t",
        fmt="%.4f",
    )

    # plot cosine similarity, std is fillbetween
    plt.figure()
    plt.plot(mean)
    plt.fill_between(
        np.arange(len(mean)),
        mean - std,
        mean + std,
        alpha=0.2,
        color=colors,
    )
    # plt.show()
    plt.savefig("cos_sim_mountain_car.png", dpi=300)


if __name__ == "__main__":
    main()
