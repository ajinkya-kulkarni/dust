from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from data import BatchTargets
from task import (
    segmentation_loss_components_per_sample,
    segmentation_loss_components_per_token,
    segmentation_loss_per_sample,
)


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
        split_head_credit: bool = True,
        token_local_credit: bool = True,
        ray_weight: float = 0.5,
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
        self.split_head_credit = split_head_credit
        self.token_local_credit = token_local_credit
        self.ray_weight = ray_weight
        self.device = next(model.parameters()).device
        if self.device.type == "mps":
            # torch.Generator(device="mps") is not supported on all PyTorch builds.
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
            # Training losses use only objectness/rays. Keep full-resolution instance
            # labels unexpanded to avoid allocating large [2K*B,256,256] tensors.
            instances=targets.instances,
            objectness=targets.objectness.repeat((repeats, 1, 1)),
            rays=targets.rays.repeat((repeats, 1, 1, 1)),
        )

    def _noise(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        kwargs = dict(device=device, dtype=torch.float32)
        if self.generator is not None:
            kwargs["generator"] = self.generator
        return torch.randn(*shape, **kwargs)

    def _uses_token_local_credit(self, site: str) -> bool:
        if not self.token_local_credit:
            return False
        local_sites = getattr(self.model, "local_credit_sites", ())
        return site in local_sites

    def _token_loss_components(
        self,
        prediction: dict[str, torch.Tensor],
        targets: BatchTargets,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return segmentation_loss_components_per_token(
            prediction,
            targets,
            token_grid=int(self.model.grid),
            local_grid=int(self.model.local_grid),
        )

    @torch.no_grad()
    def _estimate_head_component_error(
        self,
        images: torch.Tensor,
        targets: BatchTargets,
        active: torch.Tensor,
        component: str,
        scale: float,
    ) -> torch.Tensor:
        """Estimate head-output error using only one StarDist loss component."""
        clean = self.outputs["head"]
        b, t, d = clean.shape
        estimate = torch.zeros_like(clean, dtype=torch.float32)
        active_count = int(active.sum().item())
        completed = 0

        while completed < self.population:
            n = min(self.draw_chunk, self.population - completed)
            compact_noise = self._noise((n, b, t, active_count), clean.device)
            noise = torch.zeros(n, b, t, d, device=clean.device, dtype=torch.float32)
            noise[..., active] = compact_noise
            signs = torch.cat((torch.ones(n), -torch.ones(n))).to(clean.device)
            jitter = signs[:, None, None, None] * self.sigma * noise.repeat(2, 1, 1, 1)
            self.jitter_site = "head"
            self.jitter = jitter.reshape(2 * n * b, t, d)
            try:
                expanded_images = images.repeat((2 * n, 1, 1, 1))
                expanded_targets = self._repeat_targets(targets, 2 * n)
                pred = self.model(expanded_images)
                if self._uses_token_local_credit("head"):
                    obj_loss, ray_loss = self._token_loss_components(
                        pred, expanded_targets
                    )
                    component_loss = obj_loss if component == "objectness" else ray_loss
                    losses = (scale * component_loss).reshape(2, n, b, t)
                else:
                    obj_loss, ray_loss = segmentation_loss_components_per_sample(
                        pred, expanded_targets
                    )
                    component_loss = obj_loss if component == "objectness" else ray_loss
                    losses = (scale * component_loss).reshape(2, n, b)
            finally:
                self.jitter_site = None
                self.jitter = None

            directional = (losses[0] - losses[1]) / (2 * self.sigma)
            if directional.ndim == 3:
                estimate += torch.einsum(
                    "nbt,nbtd->btd", directional.float(), noise
                )
            else:
                estimate += torch.einsum(
                    "nb,nbtd->btd", directional.float(), noise
                )
            completed += n

        return estimate / self.population

    @torch.no_grad()
    def estimate_head_output_error_split(
        self,
        images: torch.Tensor,
        targets: BatchTargets,
    ) -> torch.Tensor:
        """Separate objectness and radial perturbations to avoid cross-task noise."""
        clean = self.outputs["head"]
        if clean.ndim != 3:
            raise ValueError(f"Expected [B,T,D] output at head, got {tuple(clean.shape)}")
        if not hasattr(self.model, "output_positions_per_token") or not hasattr(self.model, "n_rays"):
            raise AttributeError(
                "Split head credit requires model.output_positions_per_token and model.n_rays"
            )

        channels = 1 + int(self.model.n_rays)
        expected = int(self.model.output_positions_per_token) * channels
        if clean.shape[-1] != expected:
            raise ValueError(
                f"Head width {clean.shape[-1]} does not match "
                f"output_positions_per_token*(1+n_rays)={expected}"
            )

        indices = torch.arange(expected, device=clean.device)
        obj_active = (indices % channels) == 0
        ray_active = ~obj_active
        obj_error = self._estimate_head_component_error(
            images, targets, obj_active, "objectness", 1.0
        )
        ray_error = self._estimate_head_component_error(
            images, targets, ray_active, "rays", self.ray_weight
        )
        return obj_error + ray_error

    @torch.no_grad()
    def _estimate_token_local_output_error(
        self,
        site: str,
        images: torch.Tensor,
        targets: BatchTargets,
    ) -> torch.Tensor:
        """Estimate activation error with exact per-token downstream loss credit."""
        clean = self.outputs[site]
        if clean.ndim != 3:
            raise ValueError(f"Expected [B,T,D] output at {site}, got {tuple(clean.shape)}")
        b, t, d = clean.shape
        estimate = torch.zeros_like(clean, dtype=torch.float32)
        completed = 0

        while completed < self.population:
            n = min(self.draw_chunk, self.population - completed)
            noise = self._noise((n, b, t, d), clean.device)
            signs = torch.cat((torch.ones(n), -torch.ones(n))).to(clean.device)
            jitter = signs[:, None, None, None] * self.sigma * noise.repeat(2, 1, 1, 1)
            self.jitter_site = site
            self.jitter = jitter.reshape(2 * n * b, t, d)
            try:
                expanded_images = images.repeat((2 * n, 1, 1, 1))
                expanded_targets = self._repeat_targets(targets, 2 * n)
                pred = self.model(expanded_images)
                obj_token, ray_token = self._token_loss_components(
                    pred, expanded_targets
                )
                losses = (obj_token + self.ray_weight * ray_token).reshape(
                    2, n, b, t
                )
            finally:
                self.jitter_site = None
                self.jitter = None

            directional = (losses[0] - losses[1]) / (2 * self.sigma)
            estimate += torch.einsum(
                "nbt,nbtd->btd", directional.float(), noise
            )
            completed += n

        return estimate / self.population

    @torch.no_grad()
    def estimate_output_error(
        self,
        site: str,
        images: torch.Tensor,
        targets: BatchTargets,
    ) -> torch.Tensor:
        if site == "head" and self.split_head_credit:
            return self.estimate_head_output_error_split(images, targets)
        if self._uses_token_local_credit(site):
            return self._estimate_token_local_output_error(site, images, targets)

        clean = self.outputs[site]
        if clean.ndim != 3:
            raise ValueError(f"Expected [B,T,D] output at {site}, got {tuple(clean.shape)}")
        b, t, d = clean.shape
        estimate = torch.zeros_like(clean, dtype=torch.float32)
        completed = 0

        while completed < self.population:
            n = min(self.draw_chunk, self.population - completed)
            noise = self._noise((n, b, t, d), clean.device)
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
