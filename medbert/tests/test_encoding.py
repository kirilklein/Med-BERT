import medbert.models.get_encoding as get_encoding
import typer
from os.path import join


def main(model_dir : str = typer.Option(join("models","mlm_pretrained","test","model.pt"), help="Path to model.pt"),
    tokenized_data_path : str = typer.Option(join("data","tokenized","example_data.pt"), help="Path to tokenized data"),):
    encoder = get_encoding.Encoding(model_dir, tokenized_data_path, from_checkpoint=False)
    feat_ls = encoder.get_encoding()

if __name__ == "__main__":
    typer.run(main)