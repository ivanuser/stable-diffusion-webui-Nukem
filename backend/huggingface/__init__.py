import os.path

folder = os.path.dirname(__file__)  # huggingface
t0 = os.path.join(folder, "wan.tokenizer.json.xz")
t1 = os.path.join(folder, "Wan-AI", "Wan2.1-T2V-14B", "tokenizer", "tokenizer.json")
t2 = os.path.join(folder, "Wan-AI", "Wan2.1-I2V-14B-720P", "tokenizer", "tokenizer.json")

sha256 = "20a46ac256746594ed7e1e3ef733b83fbc5a6f0922aa7480eda961743de080ef"
# https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/blob/main/tokenizer/tokenizer.json


def decompress(source: str, target: str):
    import lzma

    with lzma.open(source, "rb") as input_file:
        with open(target, "wb") as output_file:
            output_file.write(input_file.read())


def compress(source: str, target: str):
    import lzma

    with open(source, "rb") as input_file:
        with lzma.open(target, "wb") as output_file:
            output_file.write(input_file.read())


def process():
    if not os.path.isfile(t1):
        decompress(t0, t1)
        generate_sha256(t1)
    if not os.path.isfile(t2):
        decompress(t0, t2)
        generate_sha256(t2)
    # if not os.path.isfile(t0):
    #     compress(t1, t0)


def generate_sha256(path: str):
    import hashlib

    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    assert sha256_hash.hexdigest() == sha256
