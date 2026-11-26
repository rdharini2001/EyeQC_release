"""Tests for the spatial quality maps, using synthetic fundus-like images (no datasets needed)."""
import cv2
import numpy as np
import pytest

from eyeqc.core import extract_features
from eyeqc.experiments import apply_degradation
from eyeqc.maps import MAP_KEYS, lowest_region, lowest_tile_hit, overlay, pixel_auc, quality_maps, tile_grid


def synthetic_fundus(size: int = 512, seed: int = 0) -> np.ndarray:
    """A textured orange disc on black."""
    rng = np.random.default_rng(seed)
    texture = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), 1.2)
    texture = texture / texture.std() * 22
    image = np.stack([150 + texture, 80 + texture, 40 + 0.5 * texture], axis=2)
    yy, xx = np.mgrid[:size, :size]
    disc = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 <= (0.46 * size) ** 2
    return np.clip(np.where(disc[..., None], image, 0), 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def clean_features():
    return extract_features(synthetic_fundus())


def test_maps_cover_the_retina_only(clean_features):
    maps = quality_maps(clean_features)
    inner = clean_features["inner_mask"]
    assert set(maps) == set(MAP_KEYS)
    for value in maps.values():
        assert value.shape == inner.shape
        assert np.isnan(value[~inner]).all()
        assert np.isfinite(value[inner]).all()
        assert value[inner].min() >= 0 and value[inner].max() <= 1


@pytest.mark.parametrize("defect, key", [("shadow", "exposure"), ("glare", "exposure"), ("local_blur", "detail")])
def test_map_is_lower_inside_the_defect(defect, key):
    degraded, mask = apply_degradation(synthetic_fundus(), defect, 1.0, np.random.default_rng(3))
    features = extract_features(degraded)
    score = quality_maps(features)[key]
    truth = cv2.resize(mask.astype(np.uint8), score.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    inner = features["inner_mask"]
    assert score[inner & truth].mean() < score[inner & ~truth].mean() - 0.05


def test_overlay_is_transparent_outside_the_retina(clean_features):
    image = overlay(clean_features, quality_maps(clean_features)["detail"])
    assert image.shape == (*clean_features["support_mask"].shape, 4)
    assert image[0, 0, 3] == 0 and image[256, 256, 3] == 255


def test_unscored_tiles_are_never_selected():
    grid = np.full((6, 6), np.nan)
    grid[2:4, 2:4] = 1.0
    grid[3, 3] = 0.2
    defect = np.zeros((6, 6), bool)
    defect[0, 0] = True
    assert lowest_tile_hit(grid, defect) == 0.0
    defect[3, 3] = True
    assert lowest_tile_hit(grid, defect) == 1.0


def test_tied_tiles_share_the_result():
    grid = np.full((6, 6), np.nan)
    grid[0, 1:5] = 1.0
    defect = np.zeros((6, 6), bool)
    defect[0, 4] = True
    assert lowest_tile_hit(grid, defect) == pytest.approx(0.25)


def test_tile_grid_matches_extracted_tiles(clean_features):
    grid = tile_grid(clean_features, "detail")
    assert np.isfinite(grid).sum() == len(clean_features["tiles"])


def test_pixel_auc_and_lowest_region():
    score = np.ones((40, 40))
    score[5:15, 20:30] = 0.1
    defect = np.zeros((40, 40), bool)
    defect[5:15, 20:30] = True
    assert pixel_auc(score, defect) == 1.0
    assert pixel_auc(np.ones((40, 40)), defect) == pytest.approx(0.5)
    row, column = lowest_region(score)
    assert defect[int(row), int(column)]
    assert lowest_region(np.ones((40, 40))) is None


@pytest.mark.parametrize("style", ["drop", "spectral"])
def test_overlay_styles(clean_features, style):
    image = overlay(clean_features, quality_maps(clean_features)["quality"], style=style)
    assert image.dtype == np.uint8 and image[0, 0, 3] == 0
