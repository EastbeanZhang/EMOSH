import torch

from models.utils.trainer_utils import load_checkpoint, save_checkpoint


def test_optimizer_can_continue_after_checkpoint(tmp_path):
    torch.manual_seed(4)
    model = torch.nn.Module()
    model.pipe = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(2, 3)
    optimizer.zero_grad()
    model.pipe(x).square().mean().backward()
    optimizer.step()
    save_checkpoint(tmp_path, model, optimizer, step=1, epoch=0, batch=0)
    saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
    restored = torch.nn.Module()
    restored.pipe = torch.nn.Linear(3, 2)
    resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    progress = load_checkpoint(tmp_path, restored, resumed_optimizer)
    assert progress["step"] == 1
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, saved[key])
    for current, opt in [(model, optimizer), (restored, resumed_optimizer)]:
        opt.zero_grad()
        current.pipe(x).square().mean().backward()
        opt.step()
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b)
