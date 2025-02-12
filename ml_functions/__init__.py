from .feature_engineering_functions import (
    running_mean,
    average_stats_df,
    mod_df
)

from .preprocessing_functions import (
    prepare_features,
    handle_missing_values,
    remove_outliers,
    encode_categorical_features,
    create_time_features,
    create_interaction_features
)

from .modeling_functions import (
    optimize_model,
    train_model,
    predict_proba_calibrated,
    ensemble_predictions,
    get_model_explanation
)

from .evaluation_functions import (
    evaluate_classification_metrics,
    evaluate_betting_metrics,
    evaluate_calibration,
    evaluate_feature_importance,
    evaluate_predictions_over_time
)

from .prediction_functions import (
    make_predictions,
    predict_next_games,
    predict_with_ensemble,
    predict_with_calibration,
    get_prediction_explanation,
    prepare_prediction_data
)

from .visualization_functions import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_feature_importance,
    plot_prediction_distribution,
    plot_model_comparison
)

from .metrics_functions import (
    calculate_betting_metrics,
    calculate_model_metrics,
    calculate_calibration_metrics
)

from .logging_functions import (
    setup_logger,
    log_model_performance,
    log_prediction,
    log_error
)

from .utils import (
    load_latest_data,
    save_model,
    load_model,
    format_predictions,
    calculate_stake,
    get_betting_advice
)

__version__ = '0.1.0'

__all__ = [
    # Feature Engineering
    'running_mean',
    'average_stats_df',
    'mod_df',
    
    # Preprocessing
    'prepare_features',
    'handle_missing_values',
    'remove_outliers',
    'encode_categorical_features',
    'create_time_features',
    'create_interaction_features',
    
    # Modeling
    'optimize_model',
    'train_model',
    'predict_proba_calibrated',
    'ensemble_predictions',
    'get_model_explanation',
    
    # Evaluation
    'evaluate_classification_metrics',
    'evaluate_betting_metrics',
    'evaluate_calibration',
    'evaluate_feature_importance',
    'evaluate_predictions_over_time',
    
    # Prediction
    'make_predictions',
    'predict_next_games',
    'predict_with_ensemble',
    'predict_with_calibration',
    'get_prediction_explanation',
    'prepare_prediction_data',
    
    # Visualization
    'plot_confusion_matrix',
    'plot_roc_curves',
    'plot_feature_importance',
    'plot_prediction_distribution',
    'plot_model_comparison',
    
    # Metrics
    'calculate_betting_metrics',
    'calculate_model_metrics',
    'calculate_calibration_metrics',
    
    # Logging
    'setup_logger',
    'log_model_performance',
    'log_prediction',
    'log_error',
    
    # Utils
    'load_latest_data',
    'save_model',
    'load_model',
    'format_predictions',
    'calculate_stake',
    'get_betting_advice'
] 