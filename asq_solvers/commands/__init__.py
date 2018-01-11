from allennlp.commands import main as main_allennlp


def main(prog: str = None) -> None:
    predictor_overrides = {
        "decomposable_attention": "decompatt",
        "tree_attention": "dgem"
    }
    main_allennlp(prog,
                  predictor_overrides=predictor_overrides)
