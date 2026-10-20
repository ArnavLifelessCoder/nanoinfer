# GPT2RT01 binary format

Version 1 uses little-endian unsigned integers and IEEE-754 float32 data.
The loader rejects other host byte orders. The file starts with a 64-byte header,
followed by `tensor_count` fixed 128-byte entries and then tensor data. All tensor
offsets are multiples of 64. Metadata and tensor names are copied; f32 payloads
remain read-only views into a memory mapping.

| Header offset | Type | Meaning |
| --- | --- | --- |
| 0 | 8 bytes | ASCII `GPT2RT01` |
| 8 | u32 | Version, 1 |
| 12 | 5 x u32 | Layers, model width, attention heads, vocabulary, context |
| 32 | u32 | Tensor count |
| 36 | u32 | Entry size, 128 |
| 40 | u64 | Data section offset, `64 + 128 * tensor_count` |
| 48 | u64 | Total file length including final alignment padding |
| 56 | u32 | CRC32 of header bytes 0..55 followed by the complete tensor table |
| 60 | u32 | Reserved, zero |

| Entry offset | Type | Meaning |
| --- | --- | --- |
| 0 | 64 bytes | ASCII name, null-terminated and zero-padded |
| 64 | u32 | Dtype, 1 = f32; other values rejected in M1 |
| 68 | u32 | Rank, 1..4 |
| 72 | 4 x u64 | Dimensions; unused dimensions are zero |
| 104 | u64 | Absolute tensor byte offset |
| 112 | u64 | Payload byte size, product of dimensions times 4 |
| 120 | u32 | CRC32 of the tensor payload, excluding alignment padding |
| 124 | u32 | Reserved, zero |

CRC32 is the IEEE reflected polynomial `0xedb88320`, with initial and final XOR
`0xffffffff`, matching Python `zlib.crc32`. Checksums detect accidental corruption,
not malicious replacement. Integer operations are checked before multiplication,
addition, indexing, or allocation. Names must be unique. Payloads are in table
order with only the required alignment padding, which must be zero.

Tensor names are those of `GPT2LMHeadModel.transformer`, without the `transformer.`
prefix. Conv1D matrices retain the native `[in, out]` layout for multiplication
`x @ weight`; embeddings retain `[vocabulary/context, d_model]`. `lm_head.weight`
is omitted after verifying storage identity with `wte.weight`. The exporter
prints a JSON manifest and saves the identical manifest beside the binary. It
includes every tensor's shape, offset, length and CRC32.

The mapping owner must outlive its views. Files must not be modified or truncated
while mapped. Windows denies write sharing while a mapping is open; on POSIX the
caller must ensure the same lifetime rule.

Int8 representation and scale metadata will be specified with M6. The M1 loader
must reject unsupported dtypes rather than interpreting them as f32.
