import pytest
import torch
import torch.nn.functional as F

from models.wan.modules.attention import flash_attention


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_variable_length_attention_ignores_padding(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(3)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    q = torch.randn(2, 7, 2, 16, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(2, 9, 2, 16, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    result = flash_attention(q, k, v, q_lens=torch.tensor([5, 7]), k_lens=torch.tensor([6, 9]))
    for b, (nq, nk) in enumerate([(5, 6), (7, 9)]):
        expected = F.scaled_dot_product_attention(
            q[b : b + 1, :nq].transpose(1, 2),
            k[b : b + 1, :nk].transpose(1, 2),
            v[b : b + 1, :nk].transpose(1, 2),
        ).transpose(1, 2)[0]
        torch.testing.assert_close(result[b, :nq], expected)
        assert torch.count_nonzero(result[b, nq:]) == 0
    result.float().square().mean().backward()
    assert all(torch.isfinite(x.grad).all() for x in (q, k, v))
    assert torch.count_nonzero(k.grad[0, 6:]) == 0


def test_attention_respects_query_and_softmax_scales():
    q, k, v = [torch.randn(1, 4, 2, 8) for _ in range(3)]
    out = flash_attention(q, k, v, q_scale=0.7, softmax_scale=0.2)
    expected = F.scaled_dot_product_attention(
        (q * 0.7).transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=0.2,
    ).transpose(1, 2)
    torch.testing.assert_close(out, expected)
