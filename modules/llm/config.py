def can_build(env, platform):
    # llama.cpp uses synchronous blocking calls that require JSPI/Asyncify
    # stack setup that Godot's web export doesn't provide. The WebGPU ggml
    # backend also requires async GPU init incompatible with the WASM call stack.
    if platform == "web":
        return False
    return True


def configure(env):
    pass


def get_doc_classes():
    return ["LLMModel", "LLMContext", "LLMChat"]


def get_doc_path():
    return "doc_classes"
