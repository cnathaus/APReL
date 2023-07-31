import warnings
import numpy as np

from numba import njit


class InquireLearning:
    @staticmethod
    def gradient_descent(
        feedback,
        beta,
        w_dim,
        sample_count,
        momentum=0.0,
        learning_rate=0.005,
        sample_threshold=1.0e-5,
        opt_threshold=1.0e-5,
        max_iterations=1.0e5,
        prev_w=None,
    ):
        rand = np.random.RandomState(0)
        assert sample_threshold >= opt_threshold
        selected_phis = [
            # np.array([fb.query.slate.features_matrix[fb.response]]) for fb in feedback
            np.expand_dims(
                np.array(fb.query.slate.features_matrix[fb.response]), axis=0
            )
            for fb in feedback
        ]
        # # comp_phis also contains the other option from the slate
        # comp_phis = [fb.query.slate.features_matrix[1 - fb.response] for fb in feedback]
        # comp_phis contain both options.
        comp_phis = [
            np.unique(np.array([f for f in fb.query.slate.features_matrix]), axis=0)
            for fb in feedback
        ]
        betas = [beta for fb in feedback]

        comps = comp_phis
        selections = selected_phis

        # print the lenghts of each selection and comp feature vector
        for selection in selections:
            print(np.linalg.norm(selection))

        for comp in comps:
            for opt in comp:
                print(np.linalg.norm(opt))

        opt_samples, dist_samples = [], []
        while len(opt_samples) < sample_count:
            if isinstance(prev_w, np.ndarray):
                init_w = prev_w[s]
            else:
                init_w = rand.normal(0, 1, w_dim)  # .reshape(-1,1)
            curr_w = init_w / np.linalg.norm(init_w)

            curr_diff = np.zeros_like(curr_w)
            converged = len(feedback) == 0
            diverged = len(feedback) == 0
            sample_threshold_reached = False
            it = 0
            while not converged and not diverged:
                grads = np.zeros_like(curr_w)
                for i in range(len(selections)):
                    selection_exps = np.exp(
                        betas[i] * np.dot(selections[i], curr_w)
                    ).reshape(-1, 1)
                    log_num = (betas[i] * selection_exps * selections[i]).sum(
                        axis=0
                    ) / selection_exps.sum()
                    comp_exps = np.exp(betas[i] * np.dot(comps[i], curr_w)).reshape(
                        -1, 1
                    )
                    log_denom = (betas[i] * comp_exps * comps[i]).sum(
                        axis=0
                    ) / comp_exps.sum()
                    grads = grads - (log_num - log_denom)
                    print(grads)

                new_w = curr_w - (
                    (learning_rate * np.array(grads)) + (momentum * curr_diff)
                )
                new_w = new_w / np.linalg.norm(new_w)

                if it > max_iterations:
                    warnings.warn("Maximum interations reached in Gradient Descent")
                    diverged = True
                if (
                    np.linalg.norm(new_w - curr_w) < sample_threshold
                    and not sample_threshold_reached
                ):
                    dist_samples.append(curr_w)
                    sample_threshold_reached = True
                if np.linalg.norm(new_w - curr_w) < opt_threshold:
                    converged = True
                curr_diff = new_w - curr_w
                curr_w = new_w
                it += 1
            if converged:
                opt_samples.append(curr_w)
            if len(dist_samples) < len(opt_samples):
                dist_samples.append(curr_w)
        return np.stack(dist_samples), np.stack(opt_samples)

    @staticmethod
    def numba_gradient_descent(
        feedback,
        beta,
        w_dim,
        sample_count,
        momentum=0.0,
        learning_rate=0.005,
        sample_threshold=1.0e-5,
        opt_threshold=1.0e-5,
        max_iterations=1.0e5,
        prev_w=None,
    ):
        assert sample_threshold >= opt_threshold
        selected_phis = [
            # np.array([fb.query.slate.features_matrix[fb.response]]) for fb in feedback
            np.expand_dims(
                np.array(fb.query.slate.features_matrix[fb.response]), axis=0
            )
            for fb in feedback
        ]
        # # comp_phis contains the other option from the slate # FUCK
        # comp_phis = [fb.query.slate.features_matrix[1 - fb.response] for fb in feedback]
        # comp_phis contain both options
        comp_phis = [
            np.unique(np.array([f for f in fb.query.slate.features_matrix]), axis=0)
            for fb in feedback
        ]
        betas = [beta for fb in feedback]
        comps = comp_phis
        selections = selected_phis

        # print the lenghts of each selection and comp feature vector
        # for selection in selections:
        #     print(np.linalg.norm(selection))

        # for comp in comps:
        #     for opt in comp:
        #         print(np.linalg.norm(opt))

        dist_samples, opt_samples = minimize_grads(
            betas,
            selections,
            sample_count,
            w_dim,
            len(feedback),
            learning_rate,
            momentum,
            max_iterations,
            sample_threshold,
            opt_threshold,
            comps,
            prev_w,
        )
        return np.stack(dist_samples), np.stack(opt_samples)


@njit
def minimize_grads(
    betas,
    selections,
    sample_count,
    w_dim,
    feedback_length,
    learning_rate,
    momentum,
    max_iterations,
    sample_threshold,
    opt_threshold,
    comps,
    prev_w,
):
    # test
    # prev_w = np.random.normal(
    #     0, 1, w_dim
    # )  # .reshape(-1,1) # reshape not available in numba
    opt_samples, dist_samples = [], []
    while len(opt_samples) < sample_count:
        # init_w = rand.normal(0,1,w_dim) #.reshape(-1,1)
        # if isinstance(        isinstance not available in numba
        #     prev_w, np.ndarray
        # ):  # TODO: check if performance better with prev w
        #     init_w = prev_w
        # if prev_w is not None:
        #     init_w = prev_w
        # else:
        #     init_w = np.random.normal(0, 1, w_dim)  # .reshape(-1,1)
        init_w = np.random.normal(0, 1, w_dim)  # .reshape(-1,1)
        curr_w = init_w / np.linalg.norm(init_w)
        curr_diff = np.zeros_like(curr_w)
        converged = feedback_length == 0
        diverged = feedback_length == 0
        sample_threshold_reached = False
        it = 0
        # break when converged or diverged
        while not converged and not diverged:
            grads = np.zeros_like(curr_w)
            for i in range(len(selections)):
                selection_exps = np.exp(
                    betas[i] * np.dot(selections[i], curr_w)
                ).reshape(-1, 1)
                log_num = (betas[i] * selection_exps * selections[i]).sum(
                    axis=0
                ) / selection_exps.sum()
                comp_exps = np.exp(betas[i] * np.dot(comps[i], curr_w)).reshape(-1, 1)
                log_denom = (betas[i] * comp_exps * comps[i]).sum(
                    axis=0
                ) / comp_exps.sum()
                grads = grads - (log_num - log_denom)

            new_w = curr_w - ((learning_rate * grads) + (momentum * curr_diff))
            new_w = new_w / np.linalg.norm(new_w)
            if it > max_iterations:
                print("Maximum interations reached in Gradient Descent")
                # converged = True
                diverged = True
            if (
                np.linalg.norm(new_w - curr_w) < sample_threshold
                and not sample_threshold_reached
            ):
                dist_samples.append(curr_w)
                sample_threshold_reached = True
            if np.linalg.norm(new_w - curr_w) < opt_threshold:
                converged = True
            curr_diff = new_w - curr_w
            curr_w = new_w
            it += 1
        # if converged:  # TODO: DEACTIVATE FOR REAL EXPERIMENTS
        opt_samples.append(curr_w)

        if len(dist_samples) < len(opt_samples):
            dist_samples.append(curr_w)
    return dist_samples, opt_samples
