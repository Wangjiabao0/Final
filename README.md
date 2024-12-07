# Parameters
- random_seed: A fixed seed value to ensure experiment reproducibility
- device: Specifies the computational device to use for the experiment. [cpu or 0]
- run_mode: Determines the mode of operation for the experiment. [train or test]
- train:
    - save_dir:  The root directory to save experiment outputs (e.g., logs, models)
    - model_name:  Based on this *model_name*, a unique identifier for the experiment wiil be generated, typically combining a descriptive name with a timestamp.
- test:
    - model_name: A2C
    - pth_path: The file path to the pre-trained model's .pth file to be loaded and evaluated.

