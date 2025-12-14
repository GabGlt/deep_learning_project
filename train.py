import os
from src.config import config
from src.utils import set_seed, load_alay_dict, get_logger
from src.data_loader import SimCSEDataset, load_and_split_data, create_dataloaders
from src.model import create_encoder_and_tokenizer, HateSpeechClassifier
from src.training import train_simcse, train_and_evaluate_classifier
from torch.utils.data import DataLoader


def main():
    set_seed(config.seed)
    logger = get_logger(config.logs_dir)
    logger.info(f"Using device: {config.device}")

    os.makedirs(config.models_dir, exist_ok=True)
    os.makedirs(config.logs_dir, exist_ok=True)
    os.makedirs(config.plots_dir, exist_ok=True)

    alay_dict = load_alay_dict(config.wordmap_csv_path)

    # Load, preprocess, split
    texts, labels, X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
        csv_path=config.data_csv_path,
        alay_dict=alay_dict,
        seed=config.seed,
        logger=logger,
    )

    # Tokenizer and encoder
    encoder, tokenizer = create_encoder_and_tokenizer(config.model_name)

    # Optional SimCSE pretraining
    if config.use_simcse:
        # simcse_dataset = SimCSEDataset(texts=texts, tokenizer=tokenizer, max_length=config.max_len) # Leakage potential
        simcse_dataset = SimCSEDataset(texts=X_train, tokenizer=tokenizer, max_length=config.max_len)
        simcse_loader = DataLoader(
            simcse_dataset,
            batch_size=config.batch_size_simcse,
            shuffle=True,
            num_workers=0,
        )

        logger.info("Starting SimCSE pretraining...")
        train_simcse(
            encoder=encoder,
            dataloader=simcse_loader,
            num_epochs=config.num_epochs_simcse,
            lr=config.lr_simcse,
            temperature=config.simcse_temperature,
            device=config.device,
            logger=logger,
        )

    # Supervised fine-tuning
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        tokenizer,
        config.max_len,
        config.batch_size_cls,
    )

    model = HateSpeechClassifier(encoder=encoder, num_labels=2)

    logger.info("Starting supervised fine-tuning...")
    history, metrics = train_and_evaluate_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=config.num_epochs_cls,
        lr=config.lr_cls,
        device=config.device,
        best_model_dir=config.best_model_dir,
        weight_decay=config.weight_decay,
        tokenizer=tokenizer,
        logs_dir=config.logs_dir,
        plots_dir=config.plots_dir,
        class_names=["Neutral", "Hate_Abusive"],
        logger=logger,
    )


if __name__ == "__main__":
    main()
