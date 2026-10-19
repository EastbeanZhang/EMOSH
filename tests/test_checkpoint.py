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


def test_original_constant_lr_is_saved_and_resumed(tmp_path):
    model = torch.nn.Module()
    model.pipe = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    assert optimizer.param_groups[0]["lr"] == 1e-4 / 3
    for _ in range(3):
        optimizer.zero_grad()
        model.pipe(torch.ones(1, 3)).square().mean().backward()
        optimizer.step()
        scheduler.step()
    save_checkpoint(tmp_path, model, optimizer, 3, 2, 0, scheduler)
    restored = torch.nn.Module()
    restored.pipe = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.ConstantLR(opt)
    load_checkpoint(tmp_path, restored, opt, sched)
    assert sched.state_dict() == scheduler.state_dict()
    for _ in range(3):
        for current, current_opt, current_sched in [
            (model, optimizer, scheduler),
            (restored, opt, sched),
        ]:
            current_opt.zero_grad()
            current.pipe(torch.ones(1, 3)).square().mean().backward()
            current_opt.step()
            current_sched.step()
        assert opt.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b)
    assert opt.param_groups[0]["lr"] == 1e-4
