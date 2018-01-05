from asq_solvers.commands.predict_custom import PredictCustom
from allennlp.commands import main as main_allennlp


def main(prog: str = None) -> None:
    subcommand_overrides = {
        "predict_custom": PredictCustom()
    }
    predictor_overrides = {
        "decomposable_attention": "decompatt"
    }
    main_allennlp(prog,
                  predictor_overrides=predictor_overrides,
                  subcommand_overrides=subcommand_overrides)
