# -*- coding: utf-8 -*-
"""
Module
"""

import numpy as np


def _implemented_adaptive_filters():
    """Returns a list of currently supported adaptive filters. Their names are
    equal to the names of the functions used to call the algorithms to process
    time series.
    Parameters
    ----------
        Nothing.
    Returns
    -------
    implemented_filters: list
        List of currently implemented adaptive filters.
    """
    implemented_filters = ["lms", "nlms", "apa", "rls"]
    return implemented_filters


def lms(
    desired_signals,
    input_signals,
    step_size,
    filtering_order,
    initial_weights=None,
    squeeze_outputs=True,
):
    """Least Mean Squares (LMS) method for adaptive filtering.
    This function supports multiple inputs and multiple outputs.
    Parameters
    ----------
    desired_signals: numpy.array
        An array of reference signals with shape equal to
        (sequence_length, output_dimension).
    input_signals: numpy.array
        Input signals to be filtered. The array can have shape equal to
        (sequence_length, input_dimension).
    step_size: float
        The step-size parameter for the weights updating algorithm.
    filtering_order: int
        Filter order.
    initial_weights: numpy.array, None
        Initial weigths for the filters, with shape equal to
        (output_shape, signal_dimension, order). A value of None defaults to
        an array of zeros.
        Default: None
    Returns
    -------
    error_signals: numpy.array
        Error signals for the filter, with shape equal to
        (sequence_length, output_dimension).
    output_signals: numpy.array
        Filter response signals, with shape equal to
        (sequence_length, output_dimension).
    weights: numpy.array
        Filter weights at the end of the process.
    """
    error_signals, output_signals, weights = _adaptive_filter_wrapper(
        desired_signals,
        input_signals,
        step_size,
        filtering_order,
        algorithm="lms",
        initial_weights=initial_weights,
        squeeze_outputs=squeeze_outputs,
    )

    return error_signals, output_signals, weights


def nlms(
    desired_signals,
    input_signals,
    step_size,
    filtering_order,
    epsilon=1e-5,
    initial_weights=None,
    squeeze_outputs=True,
):  # pylint: disable=too-many-arguments
    """Normalized Least Mean Squares (NLMS) method for adaptive filtering
    This function supports multiple inputs and multiple outputs.
    Parameters
    ----------
    desired_signals: numpy.array
        An array of reference signals with shape equal to
        (sequence_length, output_dimension).
    input_signals: numpy.array
        Input signals to be filtered. The array can have shape equal to
        (sequence_length, input_dimension).
    step_size: float
        The step-size parameter for the weights updating algorithm.
    filtering_order: int
        Filter order.
    epsilon: float
        Regularization parameter to avoid division by zero.
        Default: 1e-5
    initial_weights: numpy.array, None
        Initial weigths for the filters, with shape equal to
        (output_shape, signal_dimension, order). A value of None defaults to
        an array of zeros.
        Default: None
    Returns
    -------
    error_signals: numpy.array
        Error signals for the filter, with shape equal to
        (sequence_length, output_dimension).
    output_signals: numpy.array
        Filter response signals, with shape equal to
        (sequence_length, output_dimension).
    weights: numpy.array
        Filter weights at the end of the process.
    """
    error_signals, output_signals, weights = _adaptive_filter_wrapper(
        desired_signals,
        input_signals,
        step_size,
        filtering_order,
        algorithm="nlms",
        initial_weights=initial_weights,
        epsilon=epsilon,
        squeeze_outputs=squeeze_outputs,
    )

    return error_signals, output_signals, weights


def apa(
    desired_signals,
    input_signals,
    step_size,
    filtering_order,
    apa_order,
    epsilon=1e-5,
    initial_weights=None,
    squeeze_outputs=True,
):  # pylint: disable=too-many-arguments
    """Normalized Least Mean Squares (NLMS) method for adaptive filtering.
    This function supports multiple inputs and multiple outputs.
    Parameters
    ----------
    desired_signals: numpy.array
        An array of reference signals with shape equal to
        (sequence_length, output_dimension).
    input_signals: numpy.array
        Input signals to be filtered. The array can have shape equal to
        (sequence_length, input_dimension).
    step_size: float
        The step-size parameter for the weights updating algorithm.
    filtering_order: int
        Filter order.
    epsilon: float
        Regularization parameter to avoid division by zero.
        Default: 1e-5
    initial_weights: numpy.array, None
        Initial weigths for the filters, with shape equal to
        (output_shape, signal_dimension, order). A value of None defaults to
        an array of zeros.
        Default: None
    Returns
    -------
    error_signals: numpy.array
        Error signals for the filter, with shape equal to
        (sequence_length, output_dimension).
    output_signals: numpy.array
        Filter response signals, with shape equal to
        (sequence_length, output_dimension).
    weights: numpy.array
        Filter weights at the end of the process.
    """
    error_signals, output_signals, weights = _adaptive_filter_wrapper(
        desired_signals,
        input_signals,
        step_size,
        filtering_order,
        algorithm="apa",
        initial_weights=initial_weights,
        epsilon=epsilon,
        apa_order=apa_order,
        squeeze_outputs=squeeze_outputs,
    )

    return error_signals, output_signals, weights


def rls(
    desired_signals,
    input_signals,
    forgetting_factor,
    filtering_order,
    epsilon=1e-5,
    initial_weights=None,
    mode="fast",
    squeeze_outputs=True,
):  # pylint: disable=too-many-arguments
    """Recursive Least Squares (RLS) method for adaptive filtering
    This function supports multiple inputs and multiple outputs.
    Parameters
    ----------
    desired_signals: numpy.array
        An array of reference signals with shape equal to
        (sequence_length, output_dimension).
    input_signals: numpy.array
        Input signals to be filtered. The array can have shape equal to
        (sequence_length, input_dimension).
    step_size: float
        The step-size parameter for the weights updating algorithm.
    filtering_order: int
        Filter order.
    epsilon: float
        Regularization parameter to avoid division by zero.
        Default: 1e-5
    initial_weights: numpy.array, None
        Initial weigths for the filters, with shape equal to
        (output_shape, signal_dimension, order). A value of None defaults to
        an array of zeros.
        Default: None
    Returns
    -------
    error_signals: numpy.array
        Error signals for the filter, with shape equal to
        (sequence_length, output_dimension).
    output_signals: numpy.array
        Filter response signals, with shape equal to
        (sequence_length, output_dimension).
    weights: numpy.array
        Filter weights at the end of the process.
    """
    error_signals, output_signals, weights = _adaptive_filter_wrapper(
        desired_signals,
        input_signals,
        None,
        filtering_order,
        algorithm="rls",
        initial_weights=initial_weights,
        forgetting_factor=forgetting_factor,
        epsilon=epsilon,
        mode=mode,
        squeeze_outputs=squeeze_outputs,
    )

    return error_signals, output_signals, weights


def _check_adaptive_filter_specification(algorithm, **kwargs):
    """Check if the given adaptive filter is implemented and if all of its
    required parameters were passed to the wrapper function.
    Parameters
    ----------
    algorithm: string
        Indicates the adaptive algorithm to execute.
    Additional arguments:
        epsilon: float
            Regularization parameter to avoid division by zero. Used with the
            NLMS algorithm.
    Returns
    -------
        Nothing.
    """
    if algorithm not in _implemented_adaptive_filters():
        raise NotImplementedError(
            f"The {algorithm} algorithms is" " not supported."
        )
    if (algorithm == "nlms") and ("epsilon" not in kwargs):
        raise NameError(
            "Argument `epsilon` must be defined when "
            "using the NLMS algorithm."
        )
    if (algorithm == "apa") and ("epsilon" not in kwargs):
        raise NameError(
            "Argument `epsilon` must be defined when using the APA."
        )
    if (algorithm == "apa") and ("apa_order" not in kwargs):
        raise NameError(
            "Argument `apa_order` must be defined when using the APA."
        )
    if (algorithm == "rls") and ("mode" not in kwargs):
        raise NameError(
            "Argument `epsilon` must be defined when using the RLS algorithm."
        )
    if (algorithm == "rls") and ("mode" not in kwargs):
        raise NameError(
            "Argument `mode` must be defined when using the RLS algorithm."
        )
    if (algorithm == "rls") and ("forgetting_factor" not in kwargs):
        raise NameError(
            "Argument `forgetting_factor` must be defined when using"
            " the RLS algorithm."
        )
    if (algorithm == "rls") and (kwargs.get("mode") not in ["fast", "robust"]):
        raise NameError(
            "Argument `mode` must be either `fast` or `robust`. when using RLS."
        )


def _adaptive_input_signals_reshaping(desired_signals, input_signals):
    """Checks compatibility between desired signals and input signals for
    adative filtering algorithms, and reshapes these signals in a general
    form for the wrapper function.
    Parameters
    ----------
    desired_signals: numpy.ndarray
        An array of reference signals with shape equal to
        (sequence_length, output_dimension) or (sequence_length,).
    input_signals: numpy.ndarray
        Input signals to be filtered. The array can have shape equal to
        (sequence_length, input_dimension) or (sequence_length,).
    Returns
    -------
    desired_signals_reshaped: numpy.ndarray
        A reshaped array of reference signals with shape equal to
        (sequence_length, output_dimension).
    input_signals_reshaped: numpy.ndarray
        Reshaped input signals to be filtered. The array can have shape equal to
        (sequence_length, input_dimension).
    """
    if input_signals.ndim == 1:
        input_signals_reshaped = np.expand_dims(input_signals, axis=1)
    else:
        input_signals_reshaped = input_signals
    signal_length = input_signals_reshaped.shape[0]

    if desired_signals.ndim == 1:
        desired_signals_reshaped = np.expand_dims(desired_signals, axis=1)
    else:
        desired_signals_reshaped = desired_signals

    if not desired_signals_reshaped.shape[0] == signal_length:
        raise ValueError(
            "Input and desired sequences must have the same length. "
            f"Input squences have length {signal_length} and output "
            f"sequences have length {desired_signals_reshaped.shape[0]}."
        )
    return desired_signals_reshaped, input_signals_reshaped


def _adaptive_variables_initialization(
    signal_length,
    output_dimension,
    input_dimension,
    filtering_order,
    initial_weights,
    apa_order,
    algorithm,
    epsilon,
    mode,
):  # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    """Initializes variables used in the adaptive filter iterations.
    Parameters
    ----------
    signal_length: int
        Length of the input and derised sequences.
    output_dimension: int
        Dimension of the desired signal.
    input_dimension: int
        Dimension of the input signal.
    filtering_order: int
        Filter order.
    initial_weights: numpy.array, None
        Initial weigths for the filters, with shape equal to
        (output_shape, signal_dimension, order). A value of None defaults to
        an array of zeros.
        Default: None
    Returns
    -------
    weights: numpy.array
        Initial filter weights.
    output_signals: numpy.array
        Zero array to be used as the filter response, with shape
        (sequence_length, output_dimension).
    error_signals: numpy.array
        Zerro array to be used as the filter error signal, with shape
        (sequence_length, output_dimension).
    """
    if not initial_weights:  # if not specified, initialize at zero.
        weights = np.zeros(
            (output_dimension, input_dimension, filtering_order)
        )
    else:
        weights = initial_weights

    #  Algorithm variables initialization
    regressor = np.zeros(shape=(input_dimension, filtering_order))
    output_signals = np.zeros(
        shape=(
            signal_length,
            output_dimension,
        )
    )
    error_signals = np.zeros(
        shape=(
            signal_length,
            output_dimension,
        )
    )

    if algorithm != "apa":
        regression_tensor = None
        desired_samples_tensor = None
    else:
        regression_tensor = np.zeros(
            shape=(input_dimension, filtering_order, apa_order)
        )
        desired_samples_tensor = np.zeros(shape=(output_dimension, apa_order))

    if algorithm == "rls":
        if mode == "fast":
            inverse_covariance = (1.0 / epsilon) * np.einsum(
                "ab,cd->acbd", np.eye(input_dimension), np.eye(filtering_order)
            )
            covariance = None
        elif mode == "robust":
            inverse_covariance = None
            covariance = epsilon * np.einsum(
                "ab,cd->acbd", np.eye(input_dimension), np.eye(filtering_order)
            )
    else:
        inverse_covariance = None
        covariance = None

    return (
        weights,
        regressor,
        output_signals,
        error_signals,
        regression_tensor,
        desired_samples_tensor,
        inverse_covariance,
        covariance,
    )


def _adaptive_filter_wrapper(
    desired_signals,
    input_signals,
    step_size,
    filtering_order,
    algorithm,
    initial_weights=None,
    **kwargs,
):  # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    """Wrapper function that performs basic tratment of signals for their use in
    adaptive filtering algorithms and implements the basic adaptive iteration
    loop. Currently supports the Least Mean Squares (LMS) and Normalized Least
    Mean Squares Algorithm (NLMS).
    Parameters
    ----------
    desired_signals: numpy.array
        An array of reference signals with shape equal to
        (sequence_length, output_dimension).
    input_signals: numpy.array
        Input signals to be filtered. The array can have shape equal to
        (sequence_length, input_dimension).
    step_size: float
        The step-size parameter for the weights updating algorithm.
    filtering_order: int
        Filter order.
    algorithm: string
        Indicates the adaptive algorithm to execute.
        Supported values: "lms", "nlms"
    initial_weights: numpy.array, None
        Initial weigths for the filters, with shape equal to
        (output_shape, signal_dimension, order). A value of None defaults to
        an array of zeros.
        Default: None
    Additional arguments:
        epsilon: float
            Regularization parameter to avoid division by zero. Used with the
            NLMS algorithm.
    Returns
    -------
    error_signals: numpy.array
        Error signals for the filter, with shape equal to
        (sequence_length, output_dimension).
    output_signals: numpy.array
        Filter response signals, with shape equal to
        (sequence_length, output_dimension).
    weights: numpy.array
        Filter weights at the end of the process.
    """
    _check_adaptive_filter_specification(algorithm, **kwargs)

    #  Input signals reshaping and consitency verification
    (desired_signals, input_signals) = _adaptive_input_signals_reshaping(
        desired_signals, input_signals
    )
    signal_length = desired_signals.shape[0]
    output_dimension = desired_signals.shape[1]
    input_dimension = input_signals.shape[1]

    #  Initialization
    (
        weights,
        regressor,
        output_signals,
        error_signals,
        regression_tensor,
        desired_samples_tensor,
        inverse_covariance,
        covariance,
    ) = _adaptive_variables_initialization(
        signal_length,
        output_dimension,
        input_dimension,
        filtering_order,
        initial_weights,
        kwargs.get("apa_order", None),
        algorithm,
        kwargs.get("epsilon", None),
        kwargs.get("mode", None),
    )

    #  Iteration procedure
    for sample in range(signal_length):
        #  Shift regressor right and add new sample
        regressor = np.concatenate(
            [input_signals[sample : (sample + 1), :].T, regressor[:, :-1]],
            axis=1,
        )

        #  Adaptive step
        if algorithm == "lms":
            (
                error_signals[sample, :],
                output_signals[sample, :],
                weights,
            ) = _lms_step(
                desired_signals[sample, :].T,
                regressor,
                step_size,
                weights,
            )
        elif algorithm == "nlms":
            (
                error_signals[sample, :],
                output_signals[sample, :],
                weights,
            ) = _nlms_step(
                desired_signals[sample, :].T,
                regressor,
                step_size,
                kwargs["epsilon"],
                weights,
            )
        elif algorithm == "apa":
            (
                error_signals[sample, :],
                output_signals[sample, :],
                weights,
                regression_tensor,
                desired_samples_tensor,
            ) = _apa_step(
                desired_signals[sample, :].T,
                regressor,
                step_size,
                kwargs["epsilon"],
                weights,
                regression_tensor,
                desired_samples_tensor,
            )
        elif algorithm == "rls":
            if kwargs.get("mode") == "fast":
                (
                    error_signals[sample, :],
                    output_signals[sample, :],
                    weights,
                    inverse_covariance,
                ) = _rls_fast_step(
                    desired_signals[sample, :].T,
                    regressor,
                    kwargs.get("forgetting_factor"),
                    weights,
                    inverse_covariance,
                )
            elif kwargs.get("mode") == "robust":
                (
                    error_signals[sample, :],
                    output_signals[sample, :],
                    weights,
                    covariance,
                ) = _rls_robust_step(
                    desired_signals[sample, :].T,
                    regressor,
                    kwargs.get("forgetting_factor"),
                    weights,
                    covariance,
                )

    if kwargs.get("squeeze_outputs", False):
        return (
            error_signals.squeeze(),
            output_signals.squeeze(),
            weights.squeeze(),
        )
    return error_signals, output_signals, weights


def _lms_step(
    desired_samples,
    input_regressor,
    step_size,
    previous_weights,
):
    """Function that computes a step of the Least Mean Squares (LMS) filter.
    Parameters
    ----------
    desired_samples: numpy.array
        The reference signal samples for the current step, with shape equal to
        (output_shape).
    input_regressor: numpy.array
        Array with the input samples used in the current step, with shape equal
        to (signal_dimension, order).
    step_size: float
        The step-size parameter for the weights updating algorithm.
    previous_weights: numpy.array
        Previous weight array in the algorithm, with shape equal to
        (output_shape, signal_dimension, order).
    Returns
    -------
    error_samples: numpy.array
        Error signal in the current step, with shape equal to (output_shape).
    output_samples: numpy.array
        Filter response in the current step, with shape equal to (output_shape).
    weights: numpy.array
        Filter weights computed in this step, with shape equal to
        (output_shape, signal_dimension, order).
    """
    #  Compute current filter response
    output_signals_sample = np.einsum(
        "abc,bc->a", previous_weights, input_regressor
    )

    #  Compute error signal
    error_signal_sample = desired_samples - output_signals_sample

    #  Update weigths
    scaled_error = step_size * error_signal_sample
    update = np.einsum("a,bc->abc", scaled_error, input_regressor.conj())
    weights = previous_weights + update

    return (
        error_signal_sample,
        output_signals_sample,
        weights,
    )


def _nlms_step(
    desired_samples,
    input_regressor,
    step_size,
    epsilon,
    previous_weights,
):
    """Function that computes a step of the Normalized Least Mean Squares (NLMS)
     filter.
    Parameters
    ----------
    desired_samples: numpy.array
        The reference signal samples for the current step, with shape equal to
        (output_shape).
    input_regressor: numpy.array
        Array with the input samples used in the current step, with shape equal
        to (signal_dimension, order).
    step_size: float
        The step-size parameter for the weights updating algorithm.
    epsilon: float
        Regularization parameter to avoid division by zero.
    previous_weights: numpy.array
        Previous weight array in the algorithm, with shape equal to
        (output_shape, signal_dimension, order).
    Returns
    -------
    error_samples: numpy.array
        Error signal in the current step, with shape equal to (output_shape).
    output_samples: numpy.array
        Filter response in the current step, with shape equal to (output_shape).
    weights: numpy.array
        Filter weights computed in this step, with shape equal to
        (output_shape, signal_dimension, order).
    """
    #  Compute current filter response
    output_signals_sample = np.einsum(
        "abc,bc->a", previous_weights, input_regressor
    )

    #  Compute error signal
    error_signal_sample = desired_samples - output_signals_sample

    #  Normalization factor
    normalization = epsilon + np.einsum(
        "ab,ab->", input_regressor.conj(), input_regressor
    )

    #  Update weigths
    scaled_error = (step_size / normalization) * error_signal_sample
    update = np.einsum("a,bc->abc", scaled_error, input_regressor.conj())
    weights = previous_weights + update

    return (
        error_signal_sample,
        output_signals_sample,
        weights,
    )


def _apa_step(
    desired_samples,
    input_regressor,
    step_size,
    epsilon,
    previous_weights,
    regression_tensor,
    desired_samples_tensor,
):  # pylint: disable=too-many-arguments
    """Function that computes a step of the Affine Projections Algorithm (APA)
     filter.
    Parameters
    ----------
    desired_samples: numpy.ndarray
        The reference signal samples for the current step, with shape equal to
        (output_shape).
    input_regressor: numpy.ndarray
        Array with the input samples used in the current step, with shape equal
        to (signal_dimension, order).
    step_size: float
        The step-size parameter for the weights updating algorithm.
    epsilon: float
        Regularization parameter to avoid division by zero.
    previous_weights: numpy.ndarray
        Previous weight array in the algorithm, with shape equal to
        (output_shape, signal_dimension, order).
    regression_tensor: numpy.ndarray
        Previous regression tensor in the algorithm, with shape equal to
        (signal_dimension, order, apa_order).
    desired_samples_tensor: numpy.ndarray
        Tensor of previous reference smaples , with shape equal to
        (output_shape, apa_order).
    Returns
    -------
    error_samples: numpy.ndarray
        Error signal in the current step, with shape equal to (output_shape).
    output_samples: numpy.ndarray
        Filter response in the current step, with shape equal to (output_shape).
    weights: numpy.ndarray
        Filter weights computed in this step, with shape equal to
        (output_shape, signal_dimension, order).
    """
    #  Update apa-specific variables
    regression_tensor[:, :, 1:] = regression_tensor[:, :, :-1]
    regression_tensor[:, :, 0] = input_regressor

    desired_samples_tensor[:, 1:] = desired_samples_tensor[:, :-1]
    desired_samples_tensor[:, 0] = desired_samples

    #  Compute current filter response
    output_signals_sample_tensor = np.einsum(
        "abc,bcd->ad", previous_weights, regression_tensor
    )

    #  Compute error signal
    error_signal_sample_tensor = (
        desired_samples_tensor - output_signals_sample_tensor
    )  # (output_shape, apa_order)

    #  Normalization factor
    normalization_tensor = regression_tensor.shape[2] * epsilon * np.eye(
        regression_tensor.shape[2]
    ) + np.einsum("abc,abd->cd", regression_tensor.conj(), regression_tensor)

    #  Update weigths
    scaled_error = step_size * error_signal_sample_tensor
    normalized_regressor = np.reshape(
        np.linalg.solve(
            normalization_tensor,
            np.reshape(
                np.transpose(regression_tensor.conj(), (2, 0, 1)),
                (
                    regression_tensor.shape[2],
                    regression_tensor.shape[0] * regression_tensor.shape[1],
                ),
            ),
        ),
        (
            regression_tensor.shape[2],
            regression_tensor.shape[0],
            regression_tensor.shape[1],
        ),
    )  # (apa_order, signal_dimension, order)
    update = np.einsum("ab,bcd->acd", scaled_error, normalized_regressor)
    weights = previous_weights + update

    return (
        error_signal_sample_tensor[:, 0],
        output_signals_sample_tensor[:, 0],
        weights,
        regression_tensor,
        desired_samples_tensor,
    )


def _rls_fast_step(
    desired_samples,
    input_regressor,
    forgetting_factor,
    previous_weights,
    previous_inverse_covariance,
):
    """Function that computes a step of the Recursive Least Squares (APA)
     filter.
    Parameters
    ----------
    desired_samples: numpy.ndarray
        The reference signal samples for the current step, with shape equal to
        (output_shape).
    input_regressor: numpy.ndarray
        Array with the input samples used in the current step, with shape equal
        to (signal_dimension, order).
    step_size: float
        The step-size parameter for the weights updating algorithm.
    epsilon: float
        Regularization parameter to avoid division by zero.
    previous_weights: numpy.ndarray
        Previous weight array in the algorithm, with shape equal to
        (output_shape, signal_dimension, order).
    regression_tensor: numpy.ndarray
        Previous regression tensor in the algorithm, with shape equal to
        (signal_dimension, order, apa_order).
    desired_samples_tensor: numpy.ndarray
        Tensor of previous reference smaples , with shape equal to
        (output_shape, apa_order).
    Returns
    -------
    error_samples: numpy.ndarray
        Error signal in the current step, with shape equal to (output_shape).
    output_samples: numpy.ndarray
        Filter response in the current step, with shape equal to (output_shape).
    weights: numpy.ndarray
        Filter weights computed in this step, with shape equal to
        (output_shape, signal_dimension, order).
    """
    #  Compute current filter response
    output_signals_sample = np.einsum(
        "abc,bc->a", previous_weights, input_regressor
    )

    #  Compute error signal
    error_signal_sample = desired_samples - output_signals_sample

    #  Inverse covariance update
    #  See Sayed 2003, Fundamentals of Adaptive Filtering, p. 247
    inv_lambda = 1.0 / forgetting_factor
    inverse_covariance_uh = np.einsum(
        "abcd,cd->ab", previous_inverse_covariance, input_regressor.conj()
    )  # order 2 tensor
    denominator = 1.0 + inv_lambda * np.einsum(
        "ab,ab->", input_regressor, inverse_covariance_uh
    )  # sacalar
    update_term = np.einsum(
        "ab,cd->abcd",
        (inv_lambda / denominator) * inverse_covariance_uh,
        inverse_covariance_uh.conj(),
    )
    inverse_covariance_new = inv_lambda * (
        previous_inverse_covariance - update_term
    )

    #  Update weigths
    normalized_regressor = np.einsum(
        "abcd,cd->ab", inverse_covariance_new, input_regressor.conj()
    )
    update = np.einsum("a,bc->abc", error_signal_sample, normalized_regressor)
    weights = previous_weights + update

    return (
        error_signal_sample,
        output_signals_sample,
        weights,
        inverse_covariance_new,
    )


def _rls_robust_step(
    desired_samples,
    input_regressor,
    forgetting_factor,
    previous_weights,
    previous_covariance,
):
    """Function that computes a step of the Recursive Least Squares (APA)
     filter.
    Parameters
    ----------
    desired_samples: numpy.ndarray
        The reference signal samples for the current step, with shape equal to
        (output_shape).
    input_regressor: numpy.ndarray
        Array with the input samples used in the current step, with shape equal
        to (signal_dimension, order).
    step_size: float
        The step-size parameter for the weights updating algorithm.
    epsilon: float
        Regularization parameter to avoid division by zero.
    previous_weights: numpy.ndarray
        Previous weight array in the algorithm, with shape equal to
        (output_shape, signal_dimension, order).
    regression_tensor: numpy.ndarray
        Previous regression tensor in the algorithm, with shape equal to
        (signal_dimension, order, apa_order).
    desired_samples_tensor: numpy.ndarray
        Tensor of previous reference smaples , with shape equal to
        (output_shape, apa_order).
    Returns
    -------
    error_samples: numpy.ndarray
        Error signal in the current step, with shape equal to (output_shape).
    output_samples: numpy.ndarray
        Filter response in the current step, with shape equal to (output_shape).
    weights: numpy.ndarray
        Filter weights computed in this step, with shape equal to
        (output_shape, signal_dimension, order).
    """
    #  Compute current filter response
    output_signals_sample = np.einsum(
        "abc,bc->a", previous_weights, input_regressor
    )

    #  Compute error signal
    error_signal_sample = desired_samples - output_signals_sample

    #  Covariance update
    reshaped_regressor = np.reshape(
        input_regressor, (input_regressor.shape[0] * input_regressor.shape[1])
    )
    reshaped_covariance = np.reshape(
        previous_covariance,
        (
            input_regressor.shape[0] * input_regressor.shape[1],
            input_regressor.shape[0] * input_regressor.shape[1],
        ),
    )
    new_covariance = forgetting_factor * reshaped_covariance + np.einsum(
        "a,b->ab", reshaped_regressor, reshaped_regressor.conj()
    )
    #  Update weigths
    normalized_regressor = np.reshape(
        np.linalg.solve(new_covariance, reshaped_regressor.conj()),
        (input_regressor.shape[0], input_regressor.shape[1]),
    )
    update = np.einsum("a,bc->abc", error_signal_sample, normalized_regressor)
    weights = previous_weights + update

    return (
        error_signal_sample,
        output_signals_sample,
        weights,
        np.reshape(
            new_covariance,
            (
                input_regressor.shape[0],
                input_regressor.shape[1],
                input_regressor.shape[0],
                input_regressor.shape[1],
            ),
        ),
    )