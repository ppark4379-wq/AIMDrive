import pytest
import torch

from training.common.runtime.precision import causality_precision


@pytest.mark.parametrize('fail', [False, True])
def test_audit_restores_exact_parameters_buffers_and_backend_flags(fail):
    model = torch.nn.Linear(4, 2).bfloat16()
    model.register_buffer('marker', torch.tensor([1.25], dtype=torch.bfloat16))
    model.register_buffer('indices', torch.arange(3))
    parameters = list(model.parameters())
    values = {k: v.clone() for k, v in model.state_dict().items()}
    pointers = {k: v.data_ptr() for k, v in model.state_dict().items()}
    original_flag = torch.backends.cuda.matmul.allow_tf32
    try:
        with causality_precision(model):
            assert model.weight.dtype == torch.float32
            assert model.marker.dtype == torch.float32
            assert model.indices.dtype == torch.int64
            assert not torch.is_grad_enabled()
            assert not torch.backends.cuda.matmul.allow_tf32
            assert model(torch.ones(1, 4)).dtype == torch.float32
            if fail:
                raise RuntimeError('interrupted audit')
    except RuntimeError:
        assert fail
    assert all(a is b for a, b in zip(parameters, model.parameters()))
    for key, value in model.state_dict().items():
        assert value.dtype == values[key].dtype
        assert torch.equal(value, values[key])
        assert value.data_ptr() == pointers[key]
    assert torch.backends.cuda.matmul.allow_tf32 == original_flag
