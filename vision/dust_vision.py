from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from data import BatchTargets
from task import segmentation_loss_per_sample


class ForwardOnlyDUST:
    """Simple activation-perturbation DUST estimator for nn.Linear modules."""

    def __init__(
        self,
        model: nn.Module,
        sites: list[str] | None = None,
        sigma: float = 0.1,
        population: int = 64,
        draw_chunk: int = 8,
        seed: int = 123,
    ) -> None:
        self.model = model
        all_sites = OrderedDict(
            (name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)
        )
        if sites is None:
            self.modules = all_sites
        else:
            missing = [name for name in sites if name not in all_sites]
            if missing:
                raise KeyError(f"Unknown Linear sites: {missing}. Available: {list(all_sites)}")
            self.modules = OrderedDict((name, all_sites[name]) for name in sites)
        self.sigma = sigma
        self.population = population
        self.draw_chunk = min(draw_chunk, population)
        self.device = next(model.parameters()).device
        if self.device.type == "mps":
            # torch.Generator(device="mps") is not supported on all PyTorch builds.
            # Seed the global MPS RNG instead; model initialization has already happened.
            torch.manual_seed(seed)
            self.generator = None
        else:
            self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.inputs: dict[str, torch.Tensor] = {}
        self.outputs: dict[str, torch.Tensor] = {}
        self.capturing = False
        self.jitter_site: str | None = None
        self.jitter: torch.Tensor | None = None
        self.hooks = [m.register_forward_hook(self._hook(name)) for name, m in self.modules.items()]

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()

    def _hook(self, name: str):
        def hook(module, args, output):
            if self.capturing:
                self.inputs[name] = args[0].detach().float()
                self.outputs[name] = output.detach().float()
            if self.jitter_site == name:
                return output + self.jitter.to(dtype=output.dtype, device=output.device)
            return output
        return hook

    @torch.no_grad()
    def capture(self, images: torch.Tensor, targets: BatchTargets) -> torch.Tensor:
        self.inputs.clear()
        self.outputs.clear()
        self.capturing = True
        try:
            pred = self.model(images)
            loss = segmentation_loss_per_sample(pred, targets)
        finally:
            self.capturing = False
        return loss

    @staticmethod
    def _repeat_targets(targets: BatchTargets, repeats: int) -> BatchTargets:
        return BatchTargets(
            instances=targets.instances.repeat((repeats, 1, 1)),
            foreground=targets.foreground.repeat((repeats, 1, 1)),
            offsets=targets.offsets.repeat((repeats, 1, 1, 1)),
        )

    @torch.no_grad()
    def estimate_output_error(
        self,
        site: str,
        images: torch.Tensor,
        targets: BatchTargets,
    ) -> torch.Tensor:
        clean = self.outputs[site]
        if clean.ndim != 3:
            raise ValueError(f"Expected [B,T,D] output at {site}, got {tuple(clean.shape)}")
        b, t, d = clean.shape
        estimate = torch.zeros_like(clean, dtype=torch.float32)
        completed = 0

        while completed < self.population:
            n = min(self.draw_chunk, self.population - completed)
            noise_kwargs = dict(device=clean.device, dtype=torch.float32)
            if self.generator is not None:
                noise_kwargs["generator"] = self.generator
            noise = torch.randn(n, b, t, d, **noise_kwargs)
            signs = torch.cat((torch.ones(n), -torch.ones(n))).to(clean.device)
            jitter = signs[:, None, None, None] * self.sigma * noise.repeat(2, 1, 1, 1)
            self.jitter_site = site
            self.jitter = jitter.reshape(2 * n * b, t, d)
            try:
                expanded_images = images.repeat((2 * n, 1, 1, 1))
                expanded_targets = self._repeat_targets(targets, 2 * n)
                pred = self.model(expanded_images)
                losses = segmentation_loss_per_sample(pred, expanded_targets).reshape(2, n, b)
            finally:
                self.jitter_site = None
                self.jitter = None

            directional = (losses[0] - losses[1]) / (2 * self.sigma)
            estimate += torch.einsum("nb,nbtd->btd", directional.float(), noise)
            completed += n
        return estimate / self.population

    @torch.no_grad()
    def estimate_site_gradient(
        self,
        site: str,
        images: torch.Tensor,
        targets: BatchTargets,
        capture: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if capture:
            self.capture(images, targets)
        error = self.estimate_output_error(site, images, targets)
        x = self.inputs[site]
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,D] input at {site}, got {tuple(x.shape)}")
        b = x.shape[0]
        grad_w = torch.einsum("bto,bti->oi", error, x) / b
        module = self.modules[site]
        grad_b = error.sum(dim=(0, 1)) / b if module.bias is not None else None
        return grad_w, grad_b

    @torch.no_grad()
    def step(
        self,
        images: torch.Tensor,
        targets: BatchTargets,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        clean_loss = self.capture(images, targets)
        grads: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
        for site in self.modules:
            grads[site] = self.estimate_site_gradient(site, images, targets, capture=False)

        optimizer.zero_grad(set_to_none=True)
        for site, module in self.modules.items():
            grad_w, grad_b = grads[site]
            module.weight.grad = grad_w.to(dtype=module.weight.dtype)
            if module.bias is not None and grad_b is not None:
                module.bias.grad = grad_b.to(dtype=module.bias.dtype)
        optimizer.step()
        return clean_loss.mean().item()
