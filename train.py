from dataset_utils import DatasetDownloader, TrainDataset
from sklearn.model_selection import train_test_split
from augmentations.augs_creation import AugsCreation
from preprocessing.log_mel_spec import LogMelspec
from model.model import *
from train_utils.utils import *
from logger import *

import warnings

warnings.filterwarnings("ignore")

set_seed(21)


def main(config):
    writer = get_writer(config)
    dataset_downloader = DatasetDownloader(key_word)
    labeled_data, _ = dataset_downloader.generate_labeled_data()
    # create 2 dataframes for train/val so we can use augmentations only for train
    indexes = torch.randperm(len(labeled_data))
    train_indexes = indexes[:int(len(labeled_data) * 0.8)]
    val_indexes = indexes[int(len(labeled_data) * 0.8):]

    train_df = labeled_data.iloc[train_indexes].reset_index(drop=True)
    val_df = labeled_data.iloc[val_indexes].reset_index(drop=True)

    # Sample is a dict of utt, word and label
    transform_tr = AugsCreation()
    train_set = TrainDataset(df=train_df, kw=config.key_word, transform=transform_tr)
    val_set = TrainDataset(df=val_df, kw=config.key_word)

    print(f"all train({len(train_set)}) + val samples({len(val_set)}) = {len(train_set) + len(val_set)}")

    # sampler for oversampling
    train_sampler = get_sampler(train_set.df['label'].values)
    val_sampler = get_sampler(val_set.df['label'].values)

    # Dataloaders
    # Here we are obliged to use shuffle=False because of our sampler with randomness inside.

    train_loader = DataLoader(train_set, batch_size=config.batch_size,
                              shuffle=False, collate_fn=Collator(),
                              sampler=train_sampler,
                              num_workers=2, pin_memory=False)

    val_loader = DataLoader(val_set, batch_size=config.batch_size,
                            shuffle=False, collate_fn=Collator(),
                            sampler=val_sampler,
                            num_workers=2, pin_memory=False)

    melspec_train = LogMelspec(is_train=True, config=config)
    melspec_val = LogMelspec(is_train=False, config=config)

    # init model
    CRNN_model = CRNN(config)
    attn_layer = AttnMech(config)
    full_model = FullModel(config, CRNN_model, attn_layer)
    full_model = full_model.to(config.device)
    num_params, size_in_mb = get_size_in_megabytes(full_model)
    print(full_model)
    print(f"Num params: {num_params}, {count_parameters(full_model)}")
    print(f"Model size in mb: {round(size_in_mb, 3)}")
    opt = torch.optim.Adam(full_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    trainer = Trainer(writer, config)

    for n in range(config.num_epochs):
        config_writer = {
            "epoch": n + 1,
            "type": "train",
            "profile_model": False,
        }
        trainer.train_epoch(full_model, opt, train_loader, melspec_train,
                            config.gru_num_layers, config.gru_num_dirs,
                            config.hidden_size, config.device, config_writer)

        trainer.validation(full_model, val_loader, melspec_val,
                           config.gru_num_layers, config.gru_num_dirs,
                           config.hidden_size, config.device, config_writer)

        # print('END OF EPOCH', n)
        if n % 10 == 0:
            model_path = config["save_dir"] + "/" + f"model_acc_{round(trainer.get_mean_val_acc(), 2)}_epoch_{n}.pth"
            state = {
                "epoch": n,
                "state_dict": full_model.state_dict(),
                "optimizer": opt.state_dict(),
                "config": config,
            }
            torch.save(state, model_path)
            print(f"Saving checkpoint: {model_path}")


if __name__ == "__main__":
    key_word = 'sheila'  # We will use 1 key word -- 'sheila'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = {
        'verbosity': 2,
        'name': "train",
        'log_step': 50,
        'exper_name': f"kws_{key_word}",
        'key_word': key_word,
        'batch_size': 256,
        'len_epoch': 200,
        'learning_rate': 3e-4,
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'bidirectional': False,
        'n_mels': 40,  # number of mels for melspectrogram
        'kernel_size': (20, 5),  # size of kernel for convolution layer in CRNN
        'stride': (8, 2),  # size of stride for convolution layer in CRNN
        'hidden_size': 128,  # size of hidden representation in GRU
        'gru_num_layers': 2,  # number of GRU layers in CRNN
        'gru_num_dirs': 2,  # number of directions in GRU (2 if bidirectional)
        'num_classes': 2,  # number of classes (2 for "no word" or "sheila is in audio")
        'sample_rate': 16000,
        'device': device.__str__()
    }

    config = make_config(key_word, config)
    print(f"keyword: '{config.key_word}'\ndevice: {config.device}")
    main(config)
