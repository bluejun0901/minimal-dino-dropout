import copy

import pytest
import torch
from hydra import compose, initialize_config_module
from transformers import BertConfig, BertModel

from minimal_dino.config import to_train_args
from minimal_dino.evaluation import load_checkpoint
from minimal_dino.lora import LoRALinear, lora_config
from minimal_dino.model import SentenceBYOL, model_config
from minimal_dino.objective import BYOLLoss
from minimal_dino.train import (
    restore_checkpoint,
    save_checkpoint,
    set_encoder_trainable,
    update_teacher,
)


def make_model(lora=None):
    encoder = BertModel(BertConfig(
        vocab_size=32, hidden_size=12, num_hidden_layers=1,
        num_attention_heads=3, intermediate_size=24,
    ))
    return SentenceBYOL(
        encoder, projection_dim=4, projector_hidden_dim=16,
        predictor_hidden_dim=8, lora=lora,
    )


def batch():
    return dict(input_ids=torch.tensor([[1, 2, 3], [4, 5, 0]]),
                attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]))


def test_lora_starts_equivalent_and_only_updates_adapters_and_head():
    torch.manual_seed(42)
    base = make_model().eval()
    torch.manual_seed(42)
    model = make_model(dict(enabled=True, r=2, alpha=4)).eval()
    assert torch.equal(base.encode(**batch()), model.encode(**batch()))
    assert sum(isinstance(m, LoRALinear) for m in model.modules()) == 2
    set_encoder_trainable(model, False)
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    set_encoder_trainable(model, True)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    output = model(**batch(), use_dropout=False)
    output.prediction.square().sum().backward()
    for name, p in model.encoder.named_parameters():
        assert p.requires_grad == (".lora_" in name)
        assert (p.grad is not None) == (".lora_" in name)
    optimizer.step()
    changed = {name for name, p in model.named_parameters() if not torch.equal(p, before[name])}
    assert any(".lora_B" in name for name in changed)
    assert any(name.startswith("head.") for name in changed)
    assert all(".lora_" in name or name.startswith("head.") for name in changed)


def test_teacher_adapters_follow_ema():
    student = make_model(dict(enabled=True))
    teacher = copy.deepcopy(student).requires_grad_(False)
    adapter = next(m for m in student.modules() if isinstance(m, LoRALinear))
    target = next(m for m in teacher.modules() if isinstance(m, LoRALinear))
    with torch.no_grad():
        adapter.lora_B.fill_(2)
    update_teacher(student, teacher, momentum=0.75)
    torch.testing.assert_close(target.lora_B, torch.full_like(target.lora_B, 0.5))
    assert not any(p.requires_grad for p in teacher.parameters())


@pytest.mark.parametrize("enabled", [False, True])
def test_checkpoint_evaluation_reconstructs_lora(tmp_path, monkeypatch, enabled):
    model = make_model(dict(enabled=enabled)).eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.normal_()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(dict(student=model.state_dict(), head_config=model_config(model),
                    encoder_config=model.encoder.config.to_dict()), checkpoint)
    monkeypatch.setattr("minimal_dino.evaluation.AutoTokenizer.from_pretrained", lambda *a: None)
    restored, _ = load_checkpoint(checkpoint, torch.device("cpu"))
    assert model_config(restored) == model_config(model)
    torch.testing.assert_close(model.encode(**batch()), restored.encode(**batch()), rtol=0, atol=0)
    if not enabled:
        assert "lora" not in model_config(model)  # Existing checkpoints remain compatible.


def test_hydra_lora_options():
    with initialize_config_module(version_base="1.3", config_module="minimal_dino.conf"):
        config = compose(config_name="config", overrides=[
            "model.lora.enabled=true", "model.lora.r=4",
            "model.lora.target_modules=[key,value]",
        ])
    args = to_train_args(config)
    model = make_model(args.lora)
    assert isinstance(model.encoder.encoder.layer[0].attention.self.key, LoRALinear)
    assert not isinstance(model.encoder.encoder.layer[0].attention.self.query, LoRALinear)


def test_lora_training_checkpoint_resumes_optimizer(tmp_path):
    from types import SimpleNamespace

    model = make_model(dict(enabled=True))
    teacher = copy.deepcopy(model).requires_grad_(False)
    objective = BYOLLoss(embedding_dim=12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    model(**batch(), use_dropout=False).prediction.square().sum().backward()
    optimizer.step()
    scheduler.step()
    expected = copy.deepcopy(model.state_dict())
    checkpoint = save_checkpoint(
        tmp_path, model, teacher, objective, optimizer, scheduler,
        SimpleNamespace(save_pretrained=lambda path: None),
        SimpleNamespace(objective="byol"), step=1,
    )
    restored = make_model(dict(enabled=True))
    restored_teacher = copy.deepcopy(restored).requires_grad_(False)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.0)
    step = restore_checkpoint(
        checkpoint, restored, restored_teacher, objective, restored_optimizer,
        restored_scheduler, torch.device("cpu"),
    )
    assert step == 1
    assert all(torch.equal(p, expected[name]) for name, p in restored.state_dict().items())
    assert len(restored_optimizer.state) == len(optimizer.state) > 0
    set_encoder_trainable(restored, True)
    assert all(p.requires_grad == (".lora_" in name)
               for name, p in restored.encoder.named_parameters())


@pytest.mark.parametrize("config", [
    dict(r=0), dict(r=True), dict(alpha=float("nan")), dict(alpha=0),
    dict(target_modules=[]), dict(target_modules="query"), dict(enabled="true"),
])
def test_invalid_lora_options(config):
    with pytest.raises(ValueError, match="model.lora"):
        lora_config(config)


def test_unmatched_targets_fail():
    with pytest.raises(ValueError, match="did not match"):
        make_model(dict(enabled=True, target_modules=["missing"]))
