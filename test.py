from dataset_utils import DatasetDownloader, TrainDataset
from sklearn.model_selection import train_test_split
from preprocessing.log_mel_spec import LogMelspec
from model.model import *
from train_utils.utils import *
import warnings
import torch.quantization

warnings.filterwarnings("ignore")

set_seed(21)


def main(config):
    writer = None
    dataset_downloader = DatasetDownloader(key_word)
    labeled_data, _ = dataset_downloader.generate_labeled_data()
    # create 2 dataframes for train/val so we can use augmentations only for train
    train_df, val_df = train_test_split(labeled_data, test_size=0.2, stratify=labeled_data['label'], random_state=21)
    _, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    if config['multiply_val_df']:
        val_df = pd.concat([val_df] * config['multiply_val_df']).reset_index()

    val_set = TrainDataset(df=val_df, kw=config.key_word)

    print(f"test samples = {len(val_set)}")

    # sampler for oversampling
    val_sampler = get_sampler(val_set.df['label'].values)

    # Dataloaders
    # Here we are obliged to use shuffle=False because of our sampler with randomness inside.

    val_loader = DataLoader(val_set, batch_size=config.batch_size,
                            shuffle=False, collate_fn=batch_data,
                            sampler=val_sampler,
                            num_workers=2, pin_memory=False)

    melspec_val = LogMelspec(is_train=False, config=config)

    # init model
    CRNN_model = CRNN(config)
    attn_layer = AttnMech(config)
    model = FullModel(config, CRNN_model, attn_layer)

    model = model.to(config.device)

    print(model)

    checkpoint = torch.load(config["resume"])
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)

    num_params, size_in_mb = get_size_in_megabytes(model)
    print(f"Model loaded: {config['resume']}")
    print(f"Num params: {num_params}, {count_parameters(model)}")
    print(f"Model size in mb: {round(size_in_mb, 3)}")

    trainer = Trainer(writer, config)

    config_writer = {
        "type": "test",
    }

    trainer.validation(model, val_loader, melspec_val,
                       config.gru_num_layers, config.gru_num_dirs,
                       config.hidden_size, config.device, config_writer)
    if config['resume_jit']:
        model_jit = load_torchscript_model(config['resume_jit'], config.device)
        print('*' * 50)
        print('Eval jit model')
        trainer.validation(model_jit, val_loader, melspec_val,
                           config.gru_num_layers, config.gru_num_dirs,
                           config.hidden_size, config.device, config_writer)
        num_params, size_in_mb = get_size_in_megabytes(model_jit)

        print(f"Num params: {num_params}, {count_parameters(model_jit)}")
        print(f"Model size in mb: {round(size_in_mb, 3)}")

    if config["quantize_dynamic"]:
        model = model.to('cpu')
        for dict_config in config["quantize_dynamic"]:
            layers_to_quant = dict_config["layers"]
            quant_dtypes = dict_config["dtype"]
            for quant_dtype in quant_dtypes:
                print('*' * 50)
                print(quant_dtype, layers_to_quant)
                quantized_model = torch.quantization.quantize_dynamic(
                    model, layers_to_quant, dtype=quant_dtype
                )
                trainer.validation(quantized_model, val_loader, melspec_val,
                                   config.gru_num_layers, config.gru_num_dirs,
                                   config.hidden_size, config.device, config_writer)
                num_params, size_in_mb = get_size_in_megabytes(quantized_model)

                print(f"Num params: {num_params}, {count_parameters(quantized_model)}")
                print(f"Model size in mb: {round(size_in_mb, 3)}")
                # save_torchscript_model(quantized_model, "saved/models/kws_sheila/1110_220958/",
                #                       "linear_gru_int8_dynamic_quant.pt")
    if config['quantize_static']:
        backend = "fbgemm"
        model.qconfig = torch.quantization.get_default_qconfig(backend)
        model.CRNN_model.qconfig = None
        model.U.qconfig = None
        #model.attn_layer.softmax.qconfig = None
        torch.backends.quantized.engine = backend
        model_static_quantized = torch.quantization.prepare(model, inplace=True)
        model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=True)
        trainer.validation(model_static_quantized, val_loader, melspec_val,
                           config.gru_num_layers, config.gru_num_dirs,
                           config.hidden_size, config.device, config_writer)
        num_params, size_in_mb = get_size_in_megabytes(model_static_quantized)

        print(f"Num params: {num_params}")
        print(f"Model size in mb: {round(size_in_mb, 3)}")


if __name__ == "__main__":
    key_word = 'sheila'  # We will use 1 key word -- 'sheila'
    device = torch.device('cpu')  # ('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = {
        'verbosity': 2,
        'name': "test",
        'log_step': 50,
        'exper_name': f"kws_{key_word}",
        'key_word': key_word,
        'batch_size': 256,
        'len_epoch': 200,
        'learning_rate': 3e-4,
        'weight_decay': 1e-5,
        'bidirectional': False,
        'num_epochs': 100,
        'n_mels': 40,  # number of mels for melspectrogram
        'kernel_size': (20, 5),  # size of kernel for convolution layer in CRNN
        'stride': (8, 2),  # size of stride for convolution layer in CRNN
        'hidden_size': 128,  # size of hidden representation in GRU
        'gru_num_layers': 2,  # number of GRU layers in CRNN
        'gru_num_dirs': 2,  # number of directions in GRU (2 if bidirectional)
        'num_classes': 2,  # number of classes (2 for "no word" or "sheila is in audio")
        'sample_rate': 16000,
        'device': device.__str__(),
        'resume': "saved/models/kws_sheila/1113_194318/model_acc_0.96_epoch_30.pth",
        # "saved/models/kws_sheila/1110_220958/model_acc_0.97_epoch_90.pth",
        'resume_jit': False,  # "saved/models/kws_sheila/1110_220958/linear_gru_int8_dynamic_quant.pt",
        'quantize_dynamic': False, #[{"layers": {torch.nn.Linear}, "dtype": [torch.float16, torch.qint8]},
                            # {"layers": {torch.nn.Linear, nn.GRU}, "dtype": [torch.float16, torch.qint8]}],
        'quantize_static': True,
        'multiply_val_df': 5,
    }

    config = make_config(key_word, config)
    print(f"keyword: '{config.key_word}'\ndevice: {config.device}")
    main(config)
