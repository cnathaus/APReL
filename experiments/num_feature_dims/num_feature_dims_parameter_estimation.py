import aprel
import numpy as np
import gym
from aprel.querying.inquire import Inquire
from aprel.learning.inquire_learning import InquireLearning
import time
import itertools
import matplotlib.pyplot as plt
import os


def feature_func(traj):
    # the parameter estimation domain has no actions, only one state

    features = np.array(traj[0])

    return features


def find_next_queries_inquire(belief, trajectory_set):
    curr_w = np.array([sample["weights"] for sample in belief.samples])
    trajectory_indices, gains = Inquire.generate_query(trajectory_set, curr_w)
    print("trajectory_indices: ", trajectory_indices)
    query = aprel.PreferenceQuery(trajectory_set[trajectory_indices])
    queries = [query]

    return queries, gains


def update_belief_inquire(queries, responses, belief, belief_opt, true_user):
    curr_w = np.array([sample["weights"] for sample in belief.samples])
    M = len(curr_w)
    w_dim = len(curr_w[0])

    # match the responses to the queries
    queries_with_responses = []
    for i, query in enumerate(queries):
        queries_with_responses.append(aprel.Preference(query, responses[i]))
    # if queries_with_responses doesn't contain all the queries, raise an error
    if len(queries_with_responses) != len(queries):
        raise ValueError("Not all queries have been responded to.")

    t_s = time.time()
    # w_dist is for updating, w_opt is for reporting
    w_dist, w_opt = InquireLearning.numba_gradient_descent(
        queries_with_responses,
        # 1,
        20,  # from inquire run, before 1.0!!! test!
        # 0.8,
        w_dim,
        M,
        sample_threshold=0.01  # from inquire run, before 1e-5!!! test!
        # prev_w=curr_w,  # TODO: see if this improves performance
    )
    t_e = time.time()
    print("Time to update belief: ", t_e - t_s)

    # Report the metrics
    belief.samples = [{"weights": w} for w in w_dist]

    belief_opt.samples = [{"weights": w} for w in w_opt]

    return belief, belief_opt

    # if self._logging:
    #     iteration = len(self.queries)
    #     self.log_response(iteration, queries_with_responses[-1])
    #     self.log_belief(iteration)


def main():
    use_inquire = True

    num_trajectories = 1000
    num_samples = 100
    burnin = 200
    thin = 20
    proposal_distribution = aprel.gaussian_proposal

    env = aprel.Environment(feature_func)

    # run the experiment with 3, 5, 7, 9, 12 features
    features_iterations = [8, 3, 5, 7, 9, 12]
    num_iterations = 20
    num_repeats = 10

    cos_sim_list_list_list = []

    # make a directory to save the plots
    if not os.path.exists("experiment_plots"):
        os.makedirs("experiment_plots")
    if not os.path.exists("experiment_plots/parameter_estimation"):
        os.makedirs("experiment_plots/parameter_estimation")
    if not os.path.exists("experiment_plots/parameter_estimation/num_feature_dims"):
        os.makedirs("experiment_plots/parameter_estimation/num_feature_dims")

    foldername = "experiment_plots/parameter_estimation/num_feature_dims/"

    for num_features in features_iterations:
        print("##############################################")
        print("number of features: ", num_features)
        # remake the trajectory set with the new number of features
        # for parameter estimation this is just n random unit vectors with feature_dim dimensions
        trajectories = [
            aprel.TrajectoryGym(
                env=env, trajectory=[np.random.rand(num_features), None]
            )
            for _ in range(num_trajectories)
        ]

        # remake the trajectory set
        trajectory_set = aprel.TrajectorySet(trajectories)

        # Compute the query set
        subsets = np.array(
            [
                list(tup)
                for tup in itertools.combinations(np.arange(len(trajectories)), 2)
            ]
        )

        # Compute the feature differences for each query
        feature_differences = []
        colors = []

        # discard the bad trajectories
        for subset in subsets:
            feat_diff = (
                trajectories[subset[0]].features - trajectories[subset[1]].features
            )
            feature_differences.append(feat_diff)
            colors.append("b")
        feature_differences = np.array(feature_differences)

        features_dim = len(trajectory_set[0].features)

        # Initialize the query optimizer
        query_optimizer = aprel.QueryOptimizerDiscreteTrajectorySet(trajectory_set)

        cos_sim_list_list = []

        for r in range(num_repeats):
            # Initialize the object for the true human
            true_params = {
                "weights": aprel.util_funs.get_random_normalized_vector(features_dim)
            }
            true_user = aprel.SoftmaxUser(true_params)

            # Create the human response model and initialize the belief distribution
            params = {
                "weights": aprel.util_funs.get_random_normalized_vector(features_dim)
            }
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
            belief_opt = aprel.SamplingBasedBelief(
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
                    belief, belief_opt = update_belief_inquire(
                        queries, responses, belief, belief_opt, true_user
                    )
                else:
                    # Update the belief distribution
                    belief.update([aprel.Preference(query, response)])

                # Report the metrics
                print("Estimated user parameters: " + str(belief_opt.mean))
                cos_sim = aprel.cosine_similarity(belief_opt, true_user)
                print("Cosine Similarity: " + str(cos_sim))
                cos_sim_list.append(cos_sim)
            print("finished repeat: ", r + 1)
            cos_sim_list_list.append(cos_sim_list)

        cos_sim_list_list_list.append(cos_sim_list_list)

    # compute mean and std of cosine similarity for each number of features
    # i.e. 10_mean should have 20 values, ...
    # mean should have 6 lists of 20 values
    cos_sim_list_list_list = np.array(cos_sim_list_list_list)
    mean = np.mean(cos_sim_list_list_list, axis=1)
    std = np.std(cos_sim_list_list_list, axis=1)
    iterations = np.arange(num_iterations)
    # data should be lines of iterations, mean, std for each number of features
    # i.e. iterations, 10_mean, 10_std, 50_mean, 50_std, etc.
    data = np.zeros((num_iterations, 2 * len(features_iterations) + 1))
    data[:, 0] = iterations
    for i in range(len(features_iterations)):
        data[:, 2 * i + 1] = mean[i]
        data[:, 2 * i + 2] = std[i]

    # the header consists of the iteration, mean and std for each number of trajectories
    header_feature_str = [str(num) for num in features_iterations]
    header = "iteration\t" + "\t".join(
        [num_feat + "_mean\t" + num_feat + "_std" for num_feat in header_feature_str]
    )
    print(header)
    # save the mean and std to a txt file
    np.savetxt(
        foldername + "cos_sim_parameter_estimation.txt",
        data,
        header=header,
        delimiter="\t",
        fmt="%.4f",
        comments="",
    )

    # plot cosine similarity, std is fillbetween
    plt.figure()
    colors = ["b", "g", "r", "c", "m", "y"]
    for i in range(len(mean)):
        plt.plot(mean[i], label=str(features_iterations[i]), color=colors[i])
        plt.fill_between(
            np.arange(len(mean[i])),
            mean[i] - std[i],
            mean[i] + std[i],
            alpha=0.2,
            color=colors[i],
        )
        # plt.show()
        plt.savefig(
            foldername + f"cos_sim_feature_dimensions_parameter_estimation_{i}.png",
            dpi=300,
        )


if __name__ == "__main__":
    main()
