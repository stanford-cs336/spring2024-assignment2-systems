#!/usr/bin/env python3
import pytest
import torch

from .adapters import (
    get_rmsnorm_autograd_function_pytorch,
    get_rmsnorm_autograd_function_triton,
    rmsnorm_backward_g_pytorch,
    rmsnorm_backward_x_pytorch,
)


def _rmsnorm(x, g, eps=1e-5):
    r_rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * r_rms * g


def _make_rmsnorm_inputs(device=None):
    H = 50

    x = torch.randn(10, 20, H, device=device, requires_grad=True)
    g = torch.randn(H, device=device, requires_grad=True)
    # Random gradients wrt output.
    dy = 0.1 * torch.randn_like(x)

    return x, g, dy


def _test_rmsnorm_forward_pass(impl, device="cpu"):
    x, g, dy = _make_rmsnorm_inputs(device)
    y = impl(x, g)
    y_ref = _rmsnorm(x, g)

    assert torch.allclose(y, y_ref, rtol=1e-4, atol=1e-5), (y, y_ref)


def test_rmsnorm_forward_pass_pytorch():
    _test_rmsnorm_forward_pass(get_rmsnorm_autograd_function_pytorch().apply)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU must be available to run Triton kernels",
)
def test_rmsnorm_forward_pass_triton():
    _test_rmsnorm_forward_pass(
        get_rmsnorm_autograd_function_triton().apply, device="cuda"
    )


def test_rmsnorm_backward_x_pytorch():
    x, g, dy = _make_rmsnorm_inputs()
    _rmsnorm(x, g).backward(dy)
    x_grad_pred = rmsnorm_backward_x_pytorch(dy, x, g)

    assert torch.allclose(x_grad_pred, x.grad, rtol=1e-4, atol=1e-5), (
        x_grad_pred,
        x.grad,
    )


def test_rmsnorm_backward_g_pytorch():
    x, g, dy = _make_rmsnorm_inputs()
    _rmsnorm(x, g).backward(dy)
    g_grad_pred = rmsnorm_backward_g_pytorch(dy, x, g)

    assert torch.allclose(g_grad_pred, g.grad, rtol=1e-4, atol=1e-5), (
        g_grad_pred,
        g.grad,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU must be available to run Triton kernels",
)
def test_rmsnorm_backward_x_triton():
    x, g, dy = _make_rmsnorm_inputs("cuda")
    _rmsnorm(x, g).backward(dy)
    x_grad_ref = x.grad.clone()

    x.grad = None
    get_rmsnorm_autograd_function_triton().apply(x, g).backward(dy)

    assert torch.allclose(x.grad, x_grad_ref, rtol=1e-4, atol=1e-5), (
        x.grad,
        x_grad_ref,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU must be available to run Triton kernels",
)
def test_rmsnorm_backward_g_triton():
    x, g, dy = _make_rmsnorm_inputs("cuda")
    _rmsnorm(x, g).backward(dy)
    g_grad_ref = g.grad.clone()

    x.grad = None
    g.grad = None
    get_rmsnorm_autograd_function_triton().apply(x, g).backward(dy)

    assert torch.allclose(g.grad, g_grad_ref, rtol=1e-4, atol=1e-5), (
        g.grad,
        g_grad_ref,
    )


def _test_rmsnorm_forward_backward(f, device):
    x, g, dy = _make_rmsnorm_inputs(device)
    y_ref = _rmsnorm(x, g)
    y_ref.backward(dy)
    x_grad_ref = x.grad.clone()
    g_grad_ref = g.grad.clone()

    x.grad = None
    g.grad = None
    y = f(x, g)
    y.backward(dy)

    assert torch.allclose(y, y_ref, rtol=1e-4, atol=1e-5), (y, y_ref)
    assert torch.allclose(x.grad, x_grad_ref, rtol=1e-4, atol=1e-5), (
        x.grad,
        x_grad_ref,
    )
    assert torch.allclose(g.grad, g_grad_ref, rtol=1e-4, atol=1e-5), (
        g.grad,
        g_grad_ref,
    )


def test_rmsnorm_autograd_pytorch_forward_backward():
    _test_rmsnorm_forward_backward(get_rmsnorm_autograd_function_pytorch().apply, "cpu")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU must be available to run Triton kernels",
)
def test_rmsnorm_autograd_triton_forward_backward():
    _test_rmsnorm_forward_backward(get_rmsnorm_autograd_function_triton().apply, "cuda")
