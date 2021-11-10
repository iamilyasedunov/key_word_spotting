import sys

sys.path.append(sys.path[0] + "/..")

from utils.utils import *


def train_epoch(model, opt, loader, log_melspec, gru_nl, gru_nd, hidden_size, device):
    model.train()
    for i, (batch, labels) in tqdm(enumerate(loader)):
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
        acc = torch.sum(argmax_probs == labels) / torch.numel(argmax_probs)

    print(acc)


@torch.no_grad()
def validation(model, loader, log_melspec, gru_nl, gru_nd, hidden_size, device):
    model.eval()

    val_losses, accs, FAs, FRs = [], [], [], []
    all_probs, all_labels = [], []
    for i, (batch, labels) in tqdm(enumerate(loader)):
        batch, labels = batch.to(device), labels.to(device)
        batch = log_melspec(batch)

        # define frist hidden with 0
        hidden = torch.zeros(gru_nl * gru_nd, batch.size(0), hidden_size).to(device)  # (num_layers * num_dirs, BS, )
        # run model   # with autocast():
        output = model(batch, hidden)
        probs = F.softmax(output, dim=-1)  # we need probabilities so we use softmax & CE separately
        loss = F.cross_entropy(output, labels)

        # logging
        argmax_probs = torch.argmax(probs, dim=-1)
        all_probs.append(probs[:, 1].cpu())
        all_labels.append(labels.cpu())
        val_losses.append(loss.item())
        accs.append(
            torch.sum(argmax_probs == labels).item() /  # ???
            torch.numel(argmax_probs)
        )
        FA, FR = count_FA_FR(argmax_probs, labels)
        FAs.append(FA)
        FRs.append(FR)

    # area under FA/FR curve for whole loader
    au_fa_fr = get_au_fa_fr(torch.cat(all_probs, dim=0).cpu(), all_labels)
    # wandb.log({'mean_val_loss':np.mean(val_losses), 'mean_val_acc':np.mean(accs),
    #            'mean_val_FA':np.mean(FAs), 'mean_val_FR':np.mean(FRs),
    #            'au_fa_fr':au_fa_fr})
    print({'mean_val_loss': np.mean(val_losses), 'mean_val_acc': np.mean(accs),
           'mean_val_FA': np.mean(FAs), 'mean_val_FR': np.mean(FRs),
           'au_fa_fr': au_fa_fr})
    return np.mean(val_losses)
