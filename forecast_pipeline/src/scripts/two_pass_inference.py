#!/usr/bin/env python3
"""
Two-pass inference module for time series forecasting with future covariates.

This module implements the two-pass inference strategy for models that support
future covariates, particularly useful for weather forecasting applications.
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple


def perform_two_pass_inference(
    model,
    data: torch.Tensor,
    weather_covariates: Optional[Dict[str, torch.Tensor]] = None,
    future_weather_covariates: Optional[Dict[str, torch.Tensor]] = None,
    prediction_length: int = 24,
    use_future_weather: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform two-pass inference with future weather covariates.

    This function implements the two-pass inference strategy:
    1. First pass: Use available future weather covariates (up to 2 weeks)
    2. Second pass: Extend context with first pass forecast and predict remaining steps

    Args:
        model: The forecasting model with predict_quantiles method
        data: Input time series data (shape: [batch, context_length])
        weather_covariates: Past weather covariates {feature_name: tensor}
        future_weather_covariates: Future weather covariates {feature_name: tensor}
        prediction_length: Total prediction length in weeks
        use_future_weather: Whether to use future weather covariates

    Returns:
        tuple: (mean_forecast, quantile_forecast)
            - mean_forecast: Mean forecast tensor (shape: [1, prediction_length])
            - quantile_forecast: Quantile forecasts tensor (shape: [1, prediction_length, num_quantiles])
    """
    # Check if we have future covariates and should use them
    has_future_covariates = (future_weather_covariates is not None and
                            len(future_weather_covariates) > 0 and
                            use_future_weather)

    print(f"DEBUG: has_future_covariates={has_future_covariates}, weather_covariates={weather_covariates is not None}")

    if weather_covariates and has_future_covariates:
        # Check if future covariates are shorter than prediction_length
        first_future_feature = next(iter(future_weather_covariates.values()))
        future_length = len(first_future_feature)

        # Use min(2 weeks, available future weather) for Pass 1
        pass1_length = min(2, future_length)

        # Get prediction_length as integer (handle both int and mock objects)
        if hasattr(prediction_length, 'prediction_length'):
            # It's the args object, get the actual value
            prediction_length_int = prediction_length.prediction_length
        elif isinstance(prediction_length, int):
            prediction_length_int = prediction_length
        else:
            # Try to convert to int
            prediction_length_int = int(prediction_length)

        if pass1_length > 0 and pass1_length < prediction_length_int:
            # TWO-PASS INFERENCE
            # PASS 1: Predict with available future covariates
            past_covariates = {
                name: tensor.numpy()
                for name, tensor in weather_covariates.items()
            }

            # Use only the first pass1_length weeks of future covariates
            future_covariates_np = {
                name: tensor.numpy()[:pass1_length]
                for name, tensor in future_weather_covariates.items()
            }

            input_dict_pass1 = {
                "target": data.squeeze().numpy(),
                "past_covariates": past_covariates,
                "future_covariates": future_covariates_np
            }

            quantile_forecasts_pass1, mean_forecasts_pass1 = model.predict_quantiles(
                inputs=[input_dict_pass1],
                prediction_length=pass1_length
            )

            # PASS 2: Predict remaining steps using extended context
            remaining_steps = prediction_length_int - pass1_length

            # Extend target with pass 1 mean forecast
            pass1_forecast = mean_forecasts_pass1[0].squeeze()
            extended_target = np.concatenate([
                data.squeeze().numpy(),
                pass1_forecast.numpy()
            ])

            # Extend past_covariates with future_covariates from pass 1
            extended_past_covariates = {}
            for name in weather_covariates.keys():
                past_values = weather_covariates[name].numpy()
                future_values = future_weather_covariates[name].numpy()[:pass1_length]
                extended_past_covariates[name] = np.concatenate([past_values, future_values])

            input_dict_pass2 = {
                "target": extended_target,
                "past_covariates": extended_past_covariates
            }

            quantile_forecasts_pass2, mean_forecasts_pass2 = model.predict_quantiles(
                inputs=[input_dict_pass2],
                prediction_length=remaining_steps
            )

            # Concatenate forecasts
            print(f"DEBUG concat: mean_forecasts_pass1[0] shape: {mean_forecasts_pass1[0].shape}")
            print(f"DEBUG concat: mean_forecasts_pass2[0] shape: {mean_forecasts_pass2[0].shape}")
            print(f"DEBUG concat: quantile_forecasts_pass1[0] shape: {quantile_forecasts_pass1[0].shape}")
            print(f"DEBUG concat: quantile_forecasts_pass2[0] shape: {quantile_forecasts_pass2[0].shape}")
            
            mean_forecasts = [torch.cat([
                mean_forecasts_pass1[0],
                mean_forecasts_pass2[0]
            ], dim=1)]

            quantile_forecasts = [torch.cat([
                quantile_forecasts_pass1[0],
                quantile_forecasts_pass2[0]
            ], dim=1)]  # Concatenate along time dimension

        else:
            # SINGLE-PASS INFERENCE (future weather covers full prediction)
            past_covariates = {
                name: tensor.numpy()
                for name, tensor in weather_covariates.items()
            }

            # Use all available future covariates (up to prediction_length)
            future_covariates_np = {
                name: tensor.numpy()[:prediction_length]
                for name, tensor in future_weather_covariates.items()
            }

            input_dict = {
                "target": data.squeeze().numpy(),
                "past_covariates": past_covariates,
                "future_covariates": future_covariates_np
            }

            quantile_forecasts, mean_forecasts = model.predict_quantiles(
                inputs=[input_dict],
                prediction_length=prediction_length
            )

    elif weather_covariates:
        # SINGLE-PASS with only past_covariates
        past_covariates = {
            name: tensor.numpy()
            for name, tensor in weather_covariates.items()
        }

        input_dict = {
            "target": data.squeeze().numpy(),
            "past_covariates": past_covariates
        }

        quantile_forecasts, mean_forecasts = model.predict_quantiles(
            inputs=[input_dict],
            prediction_length=prediction_length
        )
        print(f"DEBUG single-pass: quantile_forecasts type: {type(quantile_forecasts)}")
        print(f"DEBUG single-pass: mean_forecasts type: {type(mean_forecasts)}")
        if hasattr(quantile_forecasts, '__len__'):
            print(f"DEBUG single-pass: quantile_forecasts[0] shape: {quantile_forecasts[0].shape}")
        if hasattr(mean_forecasts, '__len__'):
            print(f"DEBUG single-pass: mean_forecasts[0] shape: {mean_forecasts[0].shape}")
    else:
        # Traditional univariate input format
        data_3d = data.unsqueeze(-1).transpose(1, 2)

        quantile_forecasts, mean_forecasts = model.predict_quantiles(
            inputs=data_3d,
            prediction_length=prediction_length
        )

    # Debug: Check what we're actually getting from the model
    print(f"DEBUG two_pass: mean_forecasts type: {type(mean_forecasts)}, len: {len(mean_forecasts) if hasattr(mean_forecasts, '__len__') else 'N/A'}")
    print(f"DEBUG two_pass: quantile_forecasts type: {type(quantile_forecasts)}, len: {len(quantile_forecasts) if hasattr(quantile_forecasts, '__len__') else 'N/A'}")
    if hasattr(mean_forecasts, '__len__'):
        print(f"DEBUG two_pass: mean_forecasts[0] shape: {mean_forecasts[0].shape if hasattr(mean_forecasts[0], 'shape') else 'N/A'}")
    if hasattr(quantile_forecasts, '__len__'):
        print(f"DEBUG two_pass: quantile_forecasts[0] shape: {quantile_forecasts[0].shape if hasattr(quantile_forecasts[0], 'shape') else 'N/A'}")
    
    return quantile_forecasts[0], mean_forecasts[0]