import numpy as np


class Inquire:
    @staticmethod
    def gradient(feedback, w, beta):
        grads = np.zeros_like(w)
        for fb in feedback:
            phi_pos = fb.choice.selection.phi
            phis = np.array([f.phi for f in fb.choice.options])
            unique_phis = np.unique(phis, axis=0)
            exps = np.exp(beta * np.dot(unique_phis, w)).reshape(-1, 1)
            grads = grads + (
                (beta * phi_pos)
                - ((beta * exps * unique_phis).sum(axis=0) / exps.sum())
            )
        return grads * -1

    @staticmethod
    def generate_exp_mat(w_samples, trajectories, beta):
        phi = np.stack([t.features for t in trajectories])
        exp = np.exp(beta * np.dot(phi, w_samples.T))  # produces a M X N matrix
        exp_mat = np.broadcast_to(exp, (exp.shape[0], exp.shape[0], exp.shape[1]))
        return exp_mat

    @staticmethod
    def generate_prob_mat(exp):  # |Q| x |C| x |W|
        # if int_type is Modality.DEMONSTRATION:
        #     choice_matrix = np.expand_dims(np.array(list(range(exp.shape[0]))), axis=0)
        #     return np.expand_dims(exp[0] / np.sum(exp, axis=1), axis=0), choice_matrix
        # elif int_type is Modality.PREFERENCE:

        mat = exp / (exp + np.transpose(exp, (1, 0, 2)))
        idxs = np.triu_indices(exp.shape[0], 1)
        prob_mat = np.stack([mat[idxs], mat[idxs[::-1]]], axis=1)
        choices = np.transpose(np.stack(idxs))
        return prob_mat, choices

    @staticmethod
    def generate_gains_mat(prob_mat, M):
        return prob_mat * np.log(
            M * prob_mat / np.expand_dims(np.sum(prob_mat, axis=-1), axis=-1)
        )

    def generate_query(trajectories, curr_w):
        beta = 1.0
        exp_mat = Inquire.generate_exp_mat(
            curr_w, trajectories, beta
        )  # TODO: CHECK BETA VALUE
        M = len(curr_w)
        prob_mat, choice_idxs = Inquire.generate_prob_mat(exp_mat)
        gains = Inquire.generate_gains_mat(prob_mat, M)
        query_gains = np.sum(gains, axis=(1, 2)) / M

        opt_query_idx = np.argmax(query_gains)
        # query_trajs = [trajectories[i] for i in choice_idxs[opt_query_idx]]

        return choice_idxs[opt_query_idx], query_gains
