import numpy as np
import torch
from torch import linalg

# from torch.autograd.functional import jacobian
from torch.func import jacrev as jacobian


def _extended_kalman_step(
    order,
    prev_post_state,
    prev_post_cov,
    observation,
    state_transition_fcn,
    state_transition_jacobian,
    output_fcn,
    output_jacobian,
    process_cov,
    measurement_cov,
    input_sample=None,
):
    # Predict step
    state_transition_matrix = state_transition_jacobian(
        prev_post_state, input_sample
    )

    prior_state = state_transition_fcn(prev_post_state, input_sample)
    prior_cov = (
        state_transition_matrix @ prev_post_cov @ state_transition_matrix.H
        + process_cov
    )

    # Update step
    output_matrix = output_jacobian(prior_state, input_sample)

    innovation = observation - output_fcn(prior_state, input_sample)
    innovation_cov = (
        output_matrix @ prior_cov @ output_matrix.H + measurement_cov
    )
    kalman_gain = prior_cov @ output_matrix.H @ linalg.pinv(innovation_cov)
    post_state = prior_state.squeeze() + (kalman_gain @ innovation).squeeze()
    post_cov = (torch.eye(order) - kalman_gain @ output_matrix) @ prior_cov
    residual = (
        observation.squeeze() - output_fcn(post_state, input_sample).squeeze()
    )

    return post_state, post_cov, residual


def extended_kalman_identification(
    order,
    initial_state,
    initial_cov,
    tangent_flow_fcn,
    output_fcn,
    time_step,
    measurement_cov,
    observations,
    input_signals=None,
    input_derivative_signals=None,
):
    signal_length = observations.shape[0]
    num_observations = observations.shape[1]

    def state_transition_fcn(_state, _input_sample):
        new_state = _state
        rep = 10
        for _ in range(rep):
            new_state = new_state + time_step / torch.tensor(
                rep, dtype=torch.float64
            ) * tangent_flow_fcn(new_state, _input_sample)
        return new_state

    state_transition_jacobian = jacobian(state_transition_fcn, argnums=0)
    output_jacobian = jacobian(output_fcn, argnums=0)
    tangent_flow_jacobian_state = jacobian(tangent_flow_fcn, argnums=0)
    tangent_flow_jacobian_input = jacobian(tangent_flow_fcn, argnums=1)

    states = torch.zeros(size=(signal_length, order), dtype=torch.float64)
    covariances = torch.zeros(
        size=(signal_length, order, order), dtype=torch.float64
    )
    residuals = torch.zeros(
        size=(signal_length, num_observations), dtype=torch.float64
    )

    states[0, :] = torch.tensor(initial_state, dtype=torch.float64)
    covariances[0, :, :] = torch.tensor(initial_cov, dtype=torch.float64)

    if measurement_cov.ndim <= 2:
        current_measurement_cov = torch.tensor(
            measurement_cov, dtype=torch.float64
        )

    for i in range(1, signal_length):
        if input_signals is not None:
            if input_signals.ndim == 1:
                input_sample = input_signals[i - 1]
                input_derivative_sample = input_derivative_signals[i - 1]
                input_sample = torch.tensor(
                    np.array([input_sample]),
                    dtype=torch.float64,
                )
                input_derivative_sample = torch.tensor(
                    np.array([input_derivative_sample]),
                    dtype=torch.float64,
                )
            else:
                input_sample = torch.tensor(
                    input_signals[i - 1, :],
                    dtype=torch.float64,
                )
                input_derivative_sample = torch.tensor(
                    input_derivative_signals[i - 1, :],
                    dtype=torch.float64,
                )
        else:
            input_sample = None

        if observations.ndim <= 1:
            observation_sample = torch.tensor(
                observations[i - 1], dtype=torch.float64
            )
        else:
            observation_sample = torch.tensor(
                observations[i - 1, :], dtype=torch.float64
            )

        if measurement_cov.ndim == 3:
            current_measurement_cov = torch.tensor(
                measurement_cov[i - 1, :, :], dtype=torch.float64
            )

        process_cov_half = (
            torch.tensor(0.5, dtype=torch.float64)
            * torch.tensor(time_step**2, dtype=torch.float64)
            * tangent_flow_jacobian_state(states[i - 1, :], input_sample)
            @ tangent_flow_fcn(states[i - 1, :], input_sample)
        )

        if input_sample is not None:
            process_cov_half += (
                torch.tensor(0.5, dtype=torch.float64)
                * torch.tensor(time_step**2, dtype=torch.float64)
                * tangent_flow_jacobian_input(states[i - 1, :], input_sample)
                @ input_derivative_sample
            )

        if i == 1:
            process_cov = torch.einsum(
                "i,j -> ij",
                process_cov_half,
                process_cov_half.conj(),
            )
        else:
            process_cov = 0.5 * process_cov + 0.5 * torch.einsum(
                "i,j -> ij",
                process_cov_half,
                process_cov_half.conj(),
            )

        (
            states[i, :],
            covariances[i, :, :],
            residuals[i, :],
        ) = _extended_kalman_step(
            order=order,
            prev_post_state=states[i - 1, :],
            prev_post_cov=covariances[i - 1, :, :],
            observation=observation_sample,
            state_transition_fcn=state_transition_fcn,
            state_transition_jacobian=state_transition_jacobian,
            output_fcn=output_fcn,
            output_jacobian=output_jacobian,
            process_cov=process_cov,
            measurement_cov=current_measurement_cov,
            input_sample=input_sample,
        )

    return (
        states.numpy().real,
        covariances.numpy().real,
        residuals.numpy().real,
    )