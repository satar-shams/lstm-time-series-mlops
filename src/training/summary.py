class TrainingSummary:

    @staticmethod
    def print_section(title):
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)

    @staticmethod
    def default_values(DEFAULTS_PARAMS, HYPER_PARAMS)->None:
        print("\nDefault parameters:")

        no_default = True
        for key, default_value in DEFAULTS_PARAMS.items():
            if key not in HYPER_PARAMS:
                print(f"  {key} = {default_value}")
                no_default = False

        if no_default:
            print("  None (all parameters are being tuned)")

        print()

    @staticmethod
    def best_trial(study) -> None:
        print("\nBest Trial")
        print("-" * 40)
        print(f"Trial Number         : {study.best_trial.number}")
        print(f"Validation RMSE (%)  : {study.best_trial.value:.4f}")

        print("\nHyperparameters")
        print("-" * 40)
        for key, value in study.best_trial.params.items():
            print(f"{key:<20}: {value}")

    @staticmethod
    def final_summary(
        study,
        metrics: dict[str, float],
        best_run_id: str,
        model_version: int,
    ) -> None:

        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        
        best_trial = study.best_trial

        print(f"Best Trial              : {best_trial.number}")
        print(
            f"Optimization RMSE (%)   : "
            f"{best_trial.value:.4f}   "
            f"(validation — used for selection)"
        )
        print(
            f"Trial Test RMSE (%)     : "
            f"{best_trial.user_attrs['test_rmse_percent']:.4f}   "
            f"(best trial's own test score, diagnostic only)"
        )
        print(
            f"Final Test RMSE (%)     : "
            f"{metrics['rmse_percent']:.4f}   "
            f"(candidate model, retrained on train+val)"
        )
        print(
            f"Generalization Gap      : "
            f"{metrics['rmse_percent'] - best_trial.value:+.4f}   "
            f"(Final Test − Optimization RMSE)"
        )

        print("\nFinal Evaluation")
        print("-" * 60)
        print(f"MAE                   : {metrics['mae']:.4f}")
        print(f"RMSE                  : {metrics['rmse']:.4f}")
        print(f"RMSE (%)              : {metrics['rmse_percent']:.4f}")

        print("\nArtifacts")
        print("-" * 60)
        print("Production Model      : models/production_model.keras")
        print("Scaler                : models/scaler.bin")

        print("\nMLflow")
        print("-" * 60)
        print(f"Best Trial Run ID     : {best_run_id}")
        print(f"Registered Version    : v{model_version}")

        print("=" * 60)
        print("Training Completed Successfully")
        print("=" * 60)