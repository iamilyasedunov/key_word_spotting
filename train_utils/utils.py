import sys
from utils.utils import *

sys.path.append(sys.path[0] + "/..")


class Trainer():
    def __init__(self, writer, config):
        self.writer = writer
        self.config = config
        self.step = 0
        self.train_metrics = {
            "train_losses": [], "train_accs": [], "train_FAs": [], "train_FRs": [],
        }
        self.val_metrics = {
            "val_losses": [], "val_accs": [], "val_FAs": [], "val_FRs": [], "val_au_fa_fr": [],
        }

    def get_mean_val_acc(self):
        return np.mean(self.val_metrics["val_accs"])

    def train_epoch(self, model, opt, loader, log_melspec, gru_nl, gru_nd, hidden_size, device, config_writer):
        model.train()
        acc = torch.tensor([0.0])

        for i, (batch, labels) in tqdm(enumerate(loader), desc="train", total=len(loader)):
            self.step += 1

            batch, labels = batch.to(device), labels.to(device)
            batch = log_melspec(batch)

            opt.zero_grad()

            # define frist hidden with 0
            hidden = torch.zeros(gru_nl * 2, batch.size(0), hidden_size).to(device)  # (num_layers*num_dirs, BS, hidden)
            # run model # with autocast():
            logits = model(batch, hidden)
            probs = F.softmax(logits, dim=-1)  # we need probabilities so we use softmax & CE separately
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            opt.step()

            # logging
            argmax_probs = torch.argmax(probs, dim=-1)
            FA, FR = count_FA_FR(argmax_probs, labels)
            acc = torch.sum(argmax_probs == labels).item() / torch.numel(argmax_probs)

            self.train_metrics["train_losses"].append(loss.item())
            self.train_metrics["train_accs"].append(acc)
            self.train_metrics["train_FAs"].append(FA)
            self.train_metrics["train_FRs"].append(FR)

            if self.step % self.config["log_step"] == 0:
                self.writer.set_step(self.step)
                self.writer.add_scalars("train", {'loss': np.mean(self.train_metrics["train_losses"]),
                                                  'acc': np.mean(self.train_metrics["train_accs"]),
                                                  'FA': np.mean(self.train_metrics["train_FAs"]),
                                                  'FR': np.mean(self.train_metrics["train_FRs"])})
                self.train_metrics = {
                    "train_losses": [], "train_accs": [], "train_FAs": [], "train_FRs": [],
                }
        self.writer.add_scalar(f"epoch", config_writer["epoch"])

        print(f"Epoch acc: {acc}")

    @torch.no_grad()
    def validation(self, model, loader, log_melspec, gru_nl, gru_nd, hidden_size, device, config_writer):
        model.eval()

        all_probs, all_labels = [], []
        for i, (batch, labels) in tqdm(enumerate(loader), desc="val", total=len(loader)):
            batch, labels = batch.to(device), labels.to(device)
            batch = log_melspec(batch)

            # define frist hidden with 0
            hidden = torch.zeros(gru_nl * gru_nd, batch.size(0), hidden_size).to(
                device)  # (num_layers * num_dirs, BS, )
            # run model   # with autocast():
            output = model(batch, hidden)
            probs = F.softmax(output, dim=-1)  # we need probabilities so we use softmax & CE separately
            loss = F.cross_entropy(output, labels)

            # logging
            argmax_probs = torch.argmax(probs, dim=-1)
            all_probs.append(probs[:, 1].cpu())
            all_labels.append(labels.cpu())
            acc = torch.sum(argmax_probs == labels).item() / torch.numel(argmax_probs)
            FA, FR = count_FA_FR(argmax_probs, labels)

            self.val_metrics["val_losses"].append(loss.item())
            self.val_metrics["val_accs"].append(acc)
            self.val_metrics["val_FAs"].append(FA)
            self.val_metrics["val_FRs"].append(FR)
        # area under FA/FR curve for whole loader
        au_fa_fr = get_au_fa_fr(torch.cat(all_probs, dim=0).cpu(), all_labels)
        # wandb.log({'mean_val_loss':np.mean(val_losses), 'mean_val_acc':np.mean(accs),
        #            'mean_val_FA':np.mean(FAs), 'mean_val_FR':np.mean(FRs),
        #            'au_fa_fr':au_fa_fr})
        print({'mean_val_loss': np.mean(self.val_metrics["val_losses"]),
               'mean_val_acc': np.mean(self.val_metrics["val_accs"]),
               'mean_val_FA': np.mean(self.val_metrics["val_FAs"]),
               'mean_val_FR': np.mean(self.val_metrics["val_FRs"]),
               'au_fa_fr': au_fa_fr})
        self.step += 1
        self.writer.set_step(self.step, "valid")
        self.writer.add_scalars("val", {'mean_loss': np.mean(self.val_metrics["val_losses"]),
                                        'mean_acc': np.mean(self.val_metrics["val_accs"]),
                                        'mean_FA': np.mean(self.val_metrics["val_FAs"]),
                                        'mean_FR': np.mean(self.val_metrics["val_FRs"]),
                                        'au_fa_fr': au_fa_fr})

        return np.mean(self.val_metrics["val_losses"])
