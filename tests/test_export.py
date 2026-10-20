import numpy as np
import pytest
import torch

from python.export import export_weights, make_tiny, write_golden


def test_conv1d_orientation_and_tied_embedding(tmp_path):
    model = make_tiny()
    path = tmp_path / "tiny.bin"
    manifest = export_weights(model, path)
    entries = {entry["name"]: entry for entry in manifest["tensors"]}
    assert len(entries) == 4 + 12 * model.config.n_layer
    assert "lm_head.weight" not in entries
    entry = entries["h.0.attn.c_attn.weight"]
    assert entry["shape"] == [16, 48]
    exported = np.frombuffer(
        path.read_bytes(), dtype="<f4", count=16 * 48, offset=entry["offset"]
    ).reshape(16, 48)
    np.testing.assert_array_equal(
        exported, model.transformer.h[0].attn.c_attn.weight.detach().numpy()
    )
    inputs = torch.arange(32, dtype=torch.float32).reshape(2, 16)
    expected = model.transformer.h[0].attn.c_attn(inputs).detach().numpy()
    bias = model.transformer.h[0].attn.c_attn.bias.detach().numpy()
    np.testing.assert_allclose(inputs.numpy() @ exported + bias, expected, atol=1e-6)


def test_untied_output_is_rejected(tmp_path):
    model = make_tiny()
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    with pytest.raises(ValueError, match="tied"):
        export_weights(model, tmp_path / "bad.bin")


def test_unsupported_activation_is_rejected(tmp_path):
    model = make_tiny()
    model.config.activation_function = "gelu"
    with pytest.raises(ValueError, match="configuration"):
        export_weights(model, tmp_path / "bad.bin")


def test_golden_matches_direct_pytorch(tmp_path):
    torch.set_num_threads(1)
    model = make_tiny()
    write_golden(model, tmp_path, tiny=True)
    with np.load(tmp_path / "prompt_2.npz") as golden, torch.inference_mode():
        inputs = torch.tensor(golden["input_ids"].astype("int64"))[None]
        output = model(inputs, use_cache=False, output_hidden_states=True)
        np.testing.assert_array_equal(golden["logits"], output.logits[0].numpy())
        # hidden_states[1] is the output of block 0, before block 1.
        np.testing.assert_array_equal(golden["layer_0"], output.hidden_states[1][0].numpy())
        np.testing.assert_array_equal(golden["final_ln"], output.hidden_states[-1][0].numpy())
        final_from_block = model.transformer.ln_f(torch.from_numpy(golden["layer_1"])).numpy()
        np.testing.assert_array_equal(golden["final_ln"], final_from_block)
        assert golden["layer_1"].shape == (3, 16)
        assert golden["greedy_ids"].shape == (32,)
        assert golden["greedy_ids"][0] == output.logits[0, -1].argmax().item()
