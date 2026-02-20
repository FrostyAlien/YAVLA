"""Unit tests for the token merger."""

from __future__ import annotations

import torch
import pytest

from yavla.models.merger import ConcatMerger, TokenMergerConfig, merger_registry


class TestTokenMergerConfig:
    def test_defaults(self) -> None:
        cfg = TokenMergerConfig()
        assert cfg.type == "concat"
        assert cfg.num_readout_tokens == 64


class TestConcatMerger:
    def _make_merger(self, num_readout: int = 64, dim: int = 2048) -> ConcatMerger:
        return ConcatMerger(TokenMergerConfig(num_readout_tokens=num_readout), backbone_dim=dim)

    def test_output_shape(self) -> None:
        m = self._make_merger()
        B, D = 2, 2048
        img = torch.randn(B, 256, D)
        proprio = torch.randn(B, 1, D)
        lang = torch.randn(B, 20, D)
        lang_mask = torch.ones(B, 20)

        embeds, attn_mask, type_ids = m.merge(img, proprio, lang, lang_mask)
        assert embeds.shape == (B, 256 + 1 + 20 + 64, D)
        assert attn_mask.shape == (B, 341)
        assert type_ids.shape == (B, 341)

    def test_token_type_ids(self) -> None:
        m = self._make_merger()
        B, D = 2, 2048
        img = torch.randn(B, 256, D)
        proprio = torch.randn(B, 1, D)
        lang = torch.randn(B, 20, D)
        lang_mask = torch.ones(B, 20)

        _, _, type_ids = m.merge(img, proprio, lang, lang_mask)
        assert (type_ids[:, :256] == 0).all()
        assert (type_ids[:, 256:] == 1).all()

    def test_language_padding_propagation(self) -> None:
        m = self._make_merger()
        B, D = 2, 2048
        img = torch.randn(B, 256, D)
        proprio = torch.randn(B, 1, D)
        lang = torch.randn(B, 20, D)
        lang_mask = torch.ones(B, 20)
        lang_mask[:, -5:] = 0  # last 5 tokens padded

        _, attn_mask, _ = m.merge(img, proprio, lang, lang_mask)
        lang_start = 256 + 1
        assert (attn_mask[:, lang_start + 15 : lang_start + 20] == 0).all()
        assert (attn_mask[:, :lang_start] == 1).all()

    def test_readout_at_end(self) -> None:
        m = self._make_merger(num_readout=8, dim=64)
        B, D = 1, 64
        img = torch.randn(B, 4, D)
        proprio = torch.randn(B, 1, D)
        lang = torch.randn(B, 2, D)
        lang_mask = torch.ones(B, 2)

        embeds, attn_mask, _ = m.merge(img, proprio, lang, lang_mask)
        assert embeds.shape[1] == 4 + 1 + 2 + 8
        assert (attn_mask[:, -8:] == 1).all()

    def test_registry(self) -> None:
        assert "concat" in merger_registry.list()

    def test_inherits_base(self) -> None:
        from yavla.models.protocols import TokenMergerBase

        m = self._make_merger(num_readout=8, dim=64)
        assert isinstance(m, TokenMergerBase)

    def test_incomplete_merger_raises(self) -> None:
        from yavla.models.protocols import TokenMergerBase

        with pytest.raises(TypeError):
            # Missing merge() implementation
            class BadMerger(TokenMergerBase):
                pass

            BadMerger()

