from workflow.ignite.create_standard_trainer_evaluators import (
    create_standard_trainer_evaluators
)


def create_standard_trainer_validator(
    model,
    optimizer,
    train_batch,
    evaluate_batch,
    validate_data_loader,
    model_score_function,
    trainer_metrics,
    validator_metrics,
    config,
):

    trainer, evaluators = create_standard_trainer_evaluators(
        model,
        optimizer,
        train_batch,
        evaluate_batch,
        evaluate_data_loaders=dict(validate=validate_data_loader),
        model_score_function=lambda evaluators: model_score_function(evaluators['validate']),
        trainer_metrics=trainer_metrics,
        evaluator_metrics=dict(validate=validator_metrics),
        config=config,
    )

    return trainer, evaluators['validate']
