"""dynatwin — 3D BraTS2020 Segmentation + Digital Twin  v6.0"""
from dynatwin.config        import strategy, NUM_GPUS, PATCH_SIZE, NUM_CLASSES, EPOCHS, BATCH_SIZE, OUTPUT_DIR
from dynatwin.losses        import combined_seg_loss, grade_cls_loss, masked_survival_loss, SEG_METRICS, get_custom_objects
from dynatwin.data_pipeline import get_stratified_split, make_dataset, load_survival_df
from dynatwin.models        import build_unet3d_m1, build_unet3d_m2, build_unet3d_m3, DropPath
from dynatwin.evaluate      import sliding_window_inference, tta_inference, morphological_postprocess, evaluate_set
from dynatwin.visualize     import plot_training_history, plot_survival_curve, plot_metric_comparison
from dynatwin.digital_twin  import predictions_to_density, PatientDigitalTwin, PINNCalibrator, SurvivalPredictor
from dynatwin.statistics    import bootstrap_ci, wilcoxon_test, compare_models, summarise_results
from dynatwin.train         import run_one_model, WarmupCosineDecay, make_optimizer
__version__ = '6.0.0'
