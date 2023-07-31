import aprel
import numpy as np
import gym
from aprel.querying.inquire import Inquire
from aprel.learning.inquire_learning import InquireLearning
import time
import itertools
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import os
import copy


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
        sample_threshold=0.01,  # from inquire run, before 1e-5!!! test!
        max_iterations=1.0e4,
        learning_rate=0.001,
        # prev_w=curr_w,  # TODO: see if this improves performance
    )

    # w_dist, w_opt = InquireLearning.gradient_descent(
    #     queries_with_responses,
    #     # 1,
    #     20,  # from inquire run, before 1.0!!! test!
    #     # 0.8,
    #     w_dim,
    #     M,
    #     sample_threshold=0.01,  # from inquire run, before 1e-5!!! test!
    #     max_iterations=1.0e4,
    #     # prev_w=curr_w,  # TODO: see if this improves performance
    # )
    t_e = time.time()
    print("Time to update belief: ", t_e - t_s)

    # Report the metrics
    belief.samples = [{"weights": w} for w in w_dist]

    belief_opt.samples = [{"weights": w} for w in w_opt]

    return belief, belief_opt


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

    num_trajectories = 3000
    max_episode_length = 300
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

    # run the experiment with 10, 50, 100, 200, 500, 1000 trajectories
    # trajectory_iterations = [10, 50, 100, 200, 500, 1000]
    # trajectory_iterations = [10, 100, 1000]
    trajectory_iterations = [2000, 200, 20]
    # trajectory_iterations = [1000]
    # trajectory_iterations = [10, 50]
    num_iterations = 20
    num_repeats = 5

    cos_sim_list_list_list = []

    for num_traj in trajectory_iterations:
        print("##############################################")
        print("number of trajectories: ", num_traj)
        # pick a random subset of trajectories and remake the trajectory set
        trajectories = copy.deepcopy(trajectory_set.trajectories)

        np.random.shuffle(trajectories)
        trajectories = trajectories[:num_traj]
        trajectory_features = np.array(
            [trajectory.features for trajectory in trajectories]
        )

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111, projection="3d")
        ax3.scatter(
            trajectory_features[:, 0],
            trajectory_features[:, 1],
            trajectory_features[:, 2],
            c="b",
            marker="o",
            s=1,
            label="3 features",
        )
        # title
        ax3.set_title("Unnormalized Trajectory Features (with outliers)")
        # labels
        ax3.set_xlabel("Feature 1")
        ax3.set_ylabel("Feature 2")
        ax3.set_zlabel("Feature 3")

        # make plot square with equal axes, centered at (0,0)
        ax3.axis("square")

        # make a directory to save the plots
        if not os.path.exists("experiment_plots"):
            os.makedirs("experiment_plots")
        if not os.path.exists("experiment_plots/mountain_car"):
            os.makedirs("experiment_plots/mountain_car")
        if not os.path.exists("experiment_plots/mountain_car/num_queries"):
            os.makedirs("experiment_plots/mountain_car/num_queries")

        foldername = "experiment_plots/mountain_car/num_queries/"

        # save plot
        plt.savefig(
            foldername
            + "trajectory_features_test_mountain_car_unnormalized_with_outliers.png",
            dpi=300,
        )

        print(
            "number of trajectories in trajectory set before removing outliers: ",
            len(trajectories),
        )
        # discard outlier trajectories
        z_scores = np.abs(stats.zscore(trajectory_features))
        filtered_entries = (z_scores < 1).all(axis=1)
        # we need at least feature_dim trajectories to do PCA
        if np.sum(filtered_entries) < 3:
            # filter again with a higher threshold
            filtered_entries = (z_scores < 2).all(axis=1)

        trajectories = [
            trajectories[i] for i in range(len(trajectories)) if filtered_entries[i]
        ]

        trajectory_features = np.array(
            [trajectory.features for trajectory in trajectories]
        )

        print(
            "number of trajectories in trajectory set after removing outliers: ",
            len(trajectories),
        )

        # plot the trajectory features in 3D and as a histogram and save it to png

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111, projection="3d")
        ax3.scatter(
            trajectory_features[:, 0],
            trajectory_features[:, 1],
            trajectory_features[:, 2],
            c="b",
            marker="o",
            s=1,
            label="3 features",
        )
        # title
        ax3.set_title("Unnormalized Trajectory Features (no outliers)")
        # labels
        ax3.set_xlabel("Feature 1")
        ax3.set_ylabel("Feature 2")
        ax3.set_zlabel("Feature 3")

        # make plot square with equal axes, centered at (0,0)
        ax3.axis("square")

        # save plot
        plt.savefig(
            foldername
            + "trajectory_features_test_mountain_car_unnormalized_no_outliers.png",
            dpi=300,
        )

        # normalize the features
        # mean = np.mean(trajectory_features, axis=0)
        # std = np.std(trajectory_features, axis=0)
        # for trajectory in trajectories:
        #     trajectory.features = (trajectory.features - mean) / std

        # # do minmax scaling
        # scaler = MinMaxScaler(feature_range=(-1, 1))
        # trajectory_features = scaler.fit_transform(trajectory_features)

        # perform PCA
        pca = PCA(n_components=3, whiten=True)
        trajectory_features = pca.fit_transform(trajectory_features)

        # scale the features to be between -1 and 1 using minmax scaling
        scaler = MinMaxScaler(feature_range=(-1, 1))
        trajectory_features = scaler.fit_transform(trajectory_features)

        for trajectory, feature in zip(trajectories, trajectory_features):
            trajectory.features = feature

        trajectory_set_iteration = aprel.TrajectorySet(trajectories)

        trajectory_features = np.array(
            [trajectory.features for trajectory in trajectories]
        )

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

        # plot the trajectory features in 3D and as a histogram and save it to png

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection="3d")
        ax2.scatter(
            trajectory_features[:, 0],
            trajectory_features[:, 1],
            trajectory_features[:, 2],
            c="b",
            marker="o",
            s=1,
            label="3 features",
        )
        # title
        ax2.set_title("Normalized Trajectory Features (no outliers)")
        # labels
        ax2.set_xlabel("Feature 1")
        ax2.set_ylabel("Feature 2")
        ax2.set_zlabel("Feature 3")

        # make plot square with equal axes, centered at (0,0)
        ax2.axis("square")

        # save plot
        plt.savefig(
            foldername + "trajectory_features_test_mountain_car_pca_no_outliers.png",
            dpi=300,
        )

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

        # title
        ax.set_title("Feature Differences")

        # make plot square with equal axes, centered at (0,0)
        ax.axis("square")

        # save plot
        plt.savefig(foldername + "feature_differences_test_mountain_car.png", dpi=300)

        features_dim = len(trajectory_set_iteration[0].features)

        # Initialize the query optimizer
        query_optimizer = aprel.QueryOptimizerDiscreteTrajectorySet(
            trajectory_set_iteration
        )

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
                        belief, trajectory_set_iteration
                    )
                else:
                    query, objective_values = query_optimizer.optimize(
                        "mutual_information",
                        belief,
                        aprel.PreferenceQuery(trajectory_set_iteration[:2]),
                        batch_size=1,
                        optimization_method="exhaustive_search",
                        reduced_size=100,
                        gamma=1,
                        distance=aprel.default_query_distance,
                    )

                best_query_idx = np.argmax(objective_values)

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

    # compute mean and std of cosine similarity for each number of trajectories
    # i.e. 10_mean should have 20 values, ...
    # mean should have 6 lists of 20 values
    cos_sim_list_list_list = np.array(cos_sim_list_list_list)
    mean = np.mean(cos_sim_list_list_list, axis=1)
    std = np.std(cos_sim_list_list_list, axis=1)
    iterations = np.arange(num_iterations)
    # data should be lines of iterations, mean, std for each number of trajectories
    # i.e. iterations, 10_mean, 10_std, 50_mean, 50_std, etc.
    data = np.zeros((num_iterations, 2 * len(trajectory_iterations) + 1))
    data[:, 0] = iterations
    for i in range(len(trajectory_iterations)):
        data[:, 2 * i + 1] = mean[i]
        data[:, 2 * i + 2] = std[i]

    # the header consists of the iteration, mean and std for each number of trajectories
    header_traj_str = [str(num) for num in trajectory_iterations]
    header = "iteration\t" + "\t".join(
        [num_traj + "_mean\t" + num_traj + "_std" for num_traj in header_traj_str]
    )
    print(header)
    # save the mean and std to a txt file
    np.savetxt(
        foldername + "cos_sim_mountain_car.txt",
        data,
        header=header,
        delimiter="\t",
        fmt="%.4f",
        comments="",
    )

    # plot cosine similarity, std is fillbetween
    fig = plt.figure()
    colors = ["b", "g", "r", "c", "m", "y"]
    for i in range(len(mean)):
        plt.plot(mean[i], label=str(trajectory_iterations[i]), color=colors[i])
        plt.fill_between(
            np.arange(len(mean[i])),
            mean[i] - std[i],
            mean[i] + std[i],
            alpha=0.2,
            color=colors[i],
        )
        # plt.show()
        plt.savefig(foldername + f"cos_sim_mountain_car_{i}.png", dpi=300)


if __name__ == "__main__":
    main()