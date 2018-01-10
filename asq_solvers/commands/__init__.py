from asq_solvers.commands.predict_custom import PredictCustom
from allennlp.commands import main as main_allennlp


def main(prog: str = None) -> None:
    predictor_overrides = {
        "decomposable_attention": "decompatt",
    }
    subcommand_overrides = {
        "predict_custom": PredictCustom(predictor_overrides=predictor_overrides)
    }

    main_allennlp(prog,
                  predictor_overrides=predictor_overrides,
                  subcommand_overrides=subcommand_overrides)
