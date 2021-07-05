import time
# import math
import torch.nn as nn
from torch import optim
from torch.optim import Adam
from data import *
from bleu_score import *
from Architecture.model.transformer import Transformer


def count_parameters(model):
    # torch.numel(input) : returns the total number of elements in the input tensor
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def weight_initialization(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        # nn.init.kaiming_uniform(m.weight.data)
        with torch.no_grad():
            nn.init.kaiming_uniform_(m.weight)  # m.weight vs m.weight.data


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    d_ff=d_ff,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    drop_rate=drop_rate,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(weight_initialization)

optimizer = Adam(params=model.parameters(),
                 lr=initial_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=lr_decay,
                                                 patience=patience)
# src_pad_idx : integer
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)                  # =======================================

        loss = criterion(output_reshape, trg)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)  # gradient clipping
        optimizer.step()

        epoch_loss += loss.item()
        print(f'step : {round((i / len(iterator)) * 100, 2)}%, loss : {loss.item()}')

    return epoch_loss / len(iterator)  # total loss


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)              # =======================================

            loss = criterion(output_reshape, trg)

            epoch_loss += loss.item()

        # =====================================================================================

            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except ZeroDivisionError:
                    print("0으로 나눌 수 없습니다.")
                except TypeError:
                    print("타입이 맞지 않습니다.")
                except AttributeError:
                    print("잘못된 속성을 사용했습니다.")

        # =====================================================================================

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu


def run(total_epoch, best_loss):
    train_losses, valid_losses, bleus = [], [], []

    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        if step > warmup:
            """
            ReduceLROnPlateau : Decrease the learning rate when there is no performance improvement.
                                Therefore, validation loss or metric must be input as input to the step function.
                                So, when the metric does not improve, wait for the number of patience (epoch)
                                and after that, decrease the learning rate.
            """
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        valid_losses.append(valid.loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), './file/model_{}.pt'.format(valid_loss))

        f = open('./result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('./result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('./result/valid_loss.txt', 'w')
        f.write(str(valid_losses))
        f.close()

        print(f'Epoch : {step + 1} | Time : {epoch_mins}m {epoch_secs}s')
        print(f'\t Train Loss : {train_loss:.3f} | Train PPL : {math.exp(train_loss):7.3f}')
        print(f'\t Val Loss : {valid_loss:.3f} | Train PPL : {math.exp(valid_loss):7.3f}')
        print(f'\t BLEU Score : {bleu:.3f}')


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
