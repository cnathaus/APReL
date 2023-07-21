import aprel
import numpy as np
import gym
from aprel.querying.inquire import Inquire
from aprel.learning.inquire_learning import InquireLearning
import time


def feature_func(traj):
    """Returns the features of the given MountainCar trajectory, i.e. \Phi(traj).

    Args:
        traj: List of state-action tuples, e.g. [(state0, action0), (state1, action1), ...]

    Returns:
        features: a numpy vector corresponding the features of the trajectory
    """
    states = np.array([pair[0] for pair in traj])
    actions = np.array([pair[1] for pair in traj[:-1]])
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
    is_simulated_user = True
    curr_w = np.array([sample["weights"] for sample in belief.samples])
    M = len(curr_w)
    w_dim = len(curr_w[0])
    rand = np.random.RandomState(0)

    # match the responses to the queries
    queries_with_responses = []
    for query in queries:
        for response in responses:
            if query["id"] == response["id"]:
                queries_with_responses.append(
                    aprel.Preference(query["query"], response["preference"])
                )
    # if queries_with_responses doesn't contain all the queries, raise an error
    if len(queries_with_responses) != len(queries):
        raise ValueError("Not all queries have been responded to.")

    t_s = time.time()
    w_dist, w_opt = InquireLearning.gradient_descent(
        rand,
        queries_with_responses,
        1,
        w_dim,
        M,
        # prev_w=curr_w, TODO: see if this improves performance
    )
    t_e = time.time()
    print("Time to update belief: ", t_e - t_s)

    # Report the metrics
    print("Number of queries: " + str(len(queries)))
    print("Estimated user parameters: " + str(belief.mean))
    if is_simulated_user:
        cos_sim = aprel.cosine_similarity(belief, true_user)
        print("Actual user parameters: " + str(true_user.params["weights"]))
        print("Cosine Similarity: " + str(cos_sim))

    belief.samples = [{"weights": w} for w in w_opt]

    return belief

    # if self._logging:
    #     iteration = len(self.queries)
    #     self.log_response(iteration, queries_with_responses[-1])
    #     self.log_belief(iteration)


def main():
    # Create the OpenAI Gym environment
    gym_env = gym.make("MountainCarContinuous-v0")

    # Seed for reproducibility
    seed = 0
    np.random.seed(seed)
    gym_env.seed(seed)

    # Wrap the environment with a feature function
    env = aprel.GymEnvironment(gym_env, feature_func)

    num_trajectories = 10
    max_episode_length = 100
    num_iterations = 10
    num_samples = 100
    burnin = 200
    thin = 20
    proposal_distribution = aprel.gaussian_proposal

    # Create a trajectory set
    trajectory_set = aprel.generate_trajectories_randomly(
        env,
        num_trajectories=num_trajectories,
        max_episode_length=max_episode_length,
        file_name="MountainCarContinuous-v0",
        restore=False,
        headless=True,
        seed=seed,
    )
    features_dim = len(trajectory_set[0].features)

    # Initialize the query optimizer
    query_optimizer = aprel.QueryOptimizerDiscreteTrajectorySet(trajectory_set)

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

    # Initialize a dummy query so that the query optimizer will generate queries of the same kind
    query = aprel.PreferenceQuery(trajectory_set[:2])

    # Active learning loop
    for query_no in range(num_iterations):
        # Optimize the query
        # queries, objective_values = query_optimizer.optimize(
        #     args["acquisition"],
        #     belief,
        #     query,
        #     batch_size=args["batch_size"],
        #     optimization_method=args["optim_method"],
        #     reduced_size=args["reduced_size_for_batches"],
        #     gamma=args["dpp_gamma"],
        #     distance=args["distance_metric_for_batches"],
        # )
        queries, objective_values = find_next_queries_inquire(belief, trajectory_set)
        print("Objective Values: " + str(objective_values))

        # Ask the query to the human
        responses = true_user.respond(queries)

        # Update the belief distribution
        # belief.update(
        #     [
        #         aprel.Preference(query, response)
        #         for query, response in zip(queries, responses)
        #     ]
        # )

        belief = update_belief_inquire(queries, responses, belief, true_user)

        # Report the metrics
        print("Estimated user parameters: " + str(belief.mean))

        cos_sim = aprel.cosine_similarity(belief, true_user)
        print("Cosine Similarity: " + str(cos_sim))


if __name__ == "__main__":
    main()
