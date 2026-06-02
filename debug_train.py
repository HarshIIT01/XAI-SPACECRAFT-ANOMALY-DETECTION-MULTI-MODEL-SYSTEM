import traceback
from spacecraft_anomaly.config import Config
from spacecraft_anomaly.training import get_device, build_model, train_epoch, validate
from spacecraft_anomaly.data.smap_msl import SMAPMSLLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

print('A')
try:
    cfg = Config(); print('B')
    device = get_device('cpu'); print('C', device)
    loader = SMAPMSLLoader(root_dir=cfg.data.smap_msl_dir, channel='P-1', spacecraft='SMAP', window_size=128)
    print('D', loader.train_raw.shape, loader.test_raw.shape)
    train_dl, val_dl = loader.get_loaders(batch_size=64)
    print('E', len(train_dl.dataset), len(val_dl.dataset))
    n_channels = loader.n_channels; print('F', n_channels)
    model = build_model('GNN', n_channels, cfg.model, seq_len=128); print('G')
    model = model.to(device); print('H')
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=cfg.train.weight_decay); print('I')
    train_loss = train_epoch(model, train_dl, optimizer, device, 'GNN', 1, 1); print('J', train_loss)
    val_scores, val_labels = validate(model, val_dl, device); print('K', val_scores.shape, val_labels.shape)
except Exception:
    traceback.print_exc()